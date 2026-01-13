import glob
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
import uuid

import boto3
import cv2
import ffmpegcv
from google import genai
import numpy as np
import pysubs2
import requests
import torch
from tqdm import tqdm
import whisperx
from botocore.config import Config

_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v"}


def _local_path_from_key(s3_key: str) -> pathlib.Path | None:
    if s3_key.startswith("file://"):
        return pathlib.Path(s3_key[len("file://"):])
    if os.path.isabs(s3_key):
        return pathlib.Path(s3_key)
    return None


def validate_s3_key(s3_key: str) -> None:
    ext = pathlib.Path(s3_key).suffix.lower()
    if ext and ext not in _ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {ext}")


def parse_clip_moments(raw_response: str) -> list:
    cleaned_json_string = raw_response.strip()
    if cleaned_json_string.startswith("```json"):
        cleaned_json_string = cleaned_json_string[len("```json"):].strip()
    if cleaned_json_string.endswith("```"):
        cleaned_json_string = cleaned_json_string[:-len("```")].strip()

    try:
        clip_moments = json.loads(cleaned_json_string)
    except json.JSONDecodeError:
        print("Error: Identified moments is not valid JSON")
        return []

    if not clip_moments or not isinstance(clip_moments, list):
        print("Error: Identified moments is not a list")
        return []

    return clip_moments


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as exc:
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return False
        raise


def _get_s3_bucket() -> str:
    return os.getenv("OSS_BUCKET", "ai-podcast-clipper")


def _fallback_clip_moments(transcript_segments: list, min_seconds: float = 30.0, max_seconds: float = 60.0) -> list:
    if not transcript_segments:
        return []
    start = transcript_segments[0].get("start") or 0.0
    target_min = start + min_seconds
    target_max = start + max_seconds
    end = None
    for segment in transcript_segments:
        seg_end = segment.get("end")
        if seg_end is None:
            continue
        if seg_end <= target_max:
            end = seg_end
        if target_min <= seg_end <= target_max:
            end = seg_end
    if end is None:
        end = transcript_segments[-1].get("end", start)
    if end <= start:
        return []
    return [{"start": start, "end": end}]


def _get_s3_client():
    endpoint = os.getenv("OSS_ENDPOINT")
    access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
    access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    path_style = os.getenv("OSS_PATH_STYLE", "").lower() in ("1", "true", "yes")
    region = os.getenv("OSS_REGION", "us-east-1")
    signature_version = os.getenv("S3_SIGNATURE_VERSION", "s3v4")
    config = Config(
        signature_version=signature_version,
        s3={"addressing_style": "path" if path_style else "virtual"},
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=access_key_secret,
        config=config,
    )


def _call_local_llm(prompt: str) -> str:
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "").rstrip("/")
    if not base_url:
        return "[]"
    model = os.getenv("LOCAL_LLM_MODEL", "local-llama")
    api_key = os.getenv("LOCAL_LLM_API_KEY", "local")
    timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "120"))
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if not response.ok:
            print(f"Local LLM error: status={response.status_code} body={response.text[:500]}")
            return "[]"
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"Local LLM request failed: {exc}")
        return "[]"


def download_video(s3_key: str, s3_client=None) -> tuple[pathlib.Path, pathlib.Path, object]:
    validate_s3_key(s3_key)
    local_path = _local_path_from_key(s3_key)
    run_id = str(uuid.uuid4())
    base_dir = pathlib.Path("/tmp") / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    video_path = base_dir / "input.mp4"
    if local_path:
        if not local_path.exists():
            raise FileNotFoundError(f"Local video not found: {local_path}")
        shutil.copy(local_path, video_path)
        return base_dir, video_path, None

    client = s3_client or _get_s3_client()
    bucket = _get_s3_bucket()
    client.download_file(bucket, s3_key, str(video_path))
    return base_dir, video_path, client


def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, framerate=25):
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()
    print(f"[vertical] frames={len(flist)} path={pyframes_path}")

    faces = [[] for _ in range(len(flist))]

    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)

            faces[frame].append(
                {'track': tidx, 'score': avg_score, 's': track['proc_track']["s"][fidx], 'x': track['proc_track']["x"][fidx], 'y': track['proc_track']["y"][fidx]})

    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")

    vout = None
    use_gpu_writer = torch.cuda.is_available()
    for fidx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces[fidx]

        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            if use_gpu_writer:
                vout = ffmpegcv.VideoWriterNV(
                    file=temp_video_path,
                    codec=None,
                    fps=framerate,
                    resize=(target_width, target_height),
                    gpu=0,
                )
            else:
                vout = ffmpegcv.VideoWriter(
                    file=temp_video_path,
                    codec=None,
                    fps=framerate,
                    resize=(target_width, target_height),
                )

        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(
                img, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            scale_for_bg = max(
                target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_heigth = int(img.shape[0] * scale_for_bg)

            blurred_background = cv2.resize(img, (bg_width, bg_heigth))
            blurred_background = cv2.GaussianBlur(
                blurred_background, (121, 121), 0)

            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_heigth - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y +
                                                    target_height, crop_x:crop_x + target_width]

            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y +
                               resized_height, :] = resized_image

            vout.write(blurred_background)

        elif mode == "crop":
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            center_x = int(
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(min(center_x - target_width // 2,
                        frame_width - target_width), 0)

            image_cropped = resized_image[0:target_height,
                                          top_x:top_x + target_width]

            vout.write(image_cropped)

    if vout:
        vout.release()

    ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)


def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str, max_words: int = 5):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]

    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= 0:
            continue

        if not current_words:
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        elif len(current_words) >= max_words:
            subtitles.append(
                (current_start, current_end, ' '.join(current_words)))
            current_words = [word]
            current_start = start_rel
            current_end = end_rel
        else:
            current_words.append(word)
            current_end = end_rel

    if current_words:
        subtitles.append(
            (current_start, current_end, ' '.join(current_words)))

    subs = pysubs2.SSAFile()

    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 140
    new_style.primarycolor = pysubs2.Color(255, 255, 255)
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 50
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(
            start=start_time, end=end_time, text=text, style=style_name)
        subs.events.append(line)

    subs.save(subtitle_path)

    ffmpeg_cmd = (f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                  f"-c:v h264 -preset fast -crf 23 {output_path}")

    subprocess.run(ffmpeg_cmd, shell=True, check=True)


def process_clip(base_dir: pathlib.Path, original_video_path: pathlib.Path, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list, s3_client=None):
    clip_name = f"clip_{clip_index}"
    local_path = _local_path_from_key(s3_key)
    force_reprocess = os.getenv("FORCE_REPROCESS", "").lower() in ("1", "true", "yes")
    skip_render = os.getenv("SKIP_RENDER", "").lower() in ("1", "true", "yes")
    if local_path:
        output_path = local_path.parent / f"{clip_name}.mp4"
        output_s3_key = f"file://{output_path}"
    else:
        s3_key_dir = os.path.dirname(s3_key)
        output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output S3 key: {output_s3_key}")

    if not local_path:
        client = s3_client or _get_s3_client()
        bucket = _get_s3_bucket()
        if _s3_object_exists(client, bucket, output_s3_key) and not force_reprocess:
            print(f"Output already exists for {output_s3_key}, skipping processing")
            return {"output_s3_key": output_s3_key, "skipped": True}

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    if skip_render:
        print("SKIP_RENDER set; uploading raw clip segment without vertical/subtitles.")
        if local_path:
            shutil.copy(clip_segment_path, output_path)
        else:
            bucket = _get_s3_bucket()
            client.upload_file(str(clip_segment_path), bucket, output_s3_key)
        return {"output_s3_key": output_s3_key, "skipped": False, "rendered": "raw"}

    skip_face_detection = os.getenv("SKIP_FACE_DETECTION", "").lower() in ("1", "true", "yes")
    if skip_face_detection:
        print("Skipping face detection/Columbia; rendering with resize mode only.")
        frame_extract_cmd = (
            f"ffmpeg -y -i {clip_segment_path} -qscale:v 2 -r 25 "
            f"{pyframes_path}/%06d.jpg"
        )
        subprocess.run(frame_extract_cmd, shell=True, check=True, capture_output=True, text=True)
        tracks = []
        scores = []
    else:
        columbia_root = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "LR-ASD"
        columbia_script = columbia_root / "Columbia_test.py"
        columbia_command = (
            f"{sys.executable} {columbia_script} "
            f"--videoName {clip_name} "
            f"--videoFolder {str(base_dir)} "
            f"--pretrainModel {columbia_root / 'weight' / 'finetuning_TalkSet.model'}"
        )

        columbia_start_time = time.time()
        subprocess.run(columbia_command, cwd=str(columbia_root), shell=True)
        columbia_end_time = time.time()
        print(
            f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")

        tracks_path = clip_dir / "pywork" / "tracks.pckl"
        scores_path = clip_dir / "pywork" / "scores.pckl"
        if not tracks_path.exists() or not scores_path.exists():
            raise FileNotFoundError("Tracks or scores not found for clip")

        with open(tracks_path, "rb") as f:
            tracks = pickle.load(f)

        with open(scores_path, "rb") as f:
            scores = pickle.load(f)

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(
        f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

    create_subtitles_with_ffmpeg(transcript_segments, start_time,
                                 end_time, vertical_mp4_path, subtitle_output_path, max_words=5)

    if local_path:
        shutil.copy(subtitle_output_path, output_path)
    else:
        bucket = _get_s3_bucket()
        client.upload_file(
            subtitle_output_path, bucket, output_s3_key)
    return {"output_s3_key": output_s3_key, "skipped": False}


class VideoProcessor:
    def __init__(self, gemini_model: str | None = None) -> None:
        self.gemini_model = gemini_model or os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")
        self.whisperx_model_name = os.getenv("WHISPERX_MODEL", "large-v2")
        self.whisperx_model = None
        self.alignment_model = None
        self.metadata = None
        self.gemini_client = None

    def load_models(self) -> None:
        if self.whisperx_model is not None:
            return

        import torchaudio
        if not hasattr(torchaudio, "AudioMetaData"):
            try:
                from torchaudio.backend.common import AudioMetaData as _AudioMetaData
                torchaudio.AudioMetaData = _AudioMetaData
            except Exception:
                class AudioMetaData:
                    def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
                        self.sample_rate = sample_rate
                        self.num_frames = num_frames
                        self.num_channels = num_channels
                        self.bits_per_sample = bits_per_sample
                        self.encoding = encoding

                torchaudio.AudioMetaData = AudioMetaData

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        if not hasattr(torchaudio, "info"):
            try:
                import soundfile as sf
            except Exception:
                def info(*_args, **_kwargs):
                    raise RuntimeError("torchaudio.info is unavailable and soundfile is not installed.")
            else:
                def info(file, backend=None):
                    with sf.SoundFile(file) as fh:
                        return torchaudio.AudioMetaData(
                            sample_rate=fh.samplerate,
                            num_frames=len(fh),
                            num_channels=fh.channels,
                            bits_per_sample=0,
                            encoding="UNKNOWN",
                        )
            torchaudio.info = info

        print("Loading models")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"

        try:
            import typing
            from omegaconf import DictConfig, ListConfig
            from omegaconf.base import ContainerMetadata
            torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, typing.Any])
        except Exception:
            pass

        if not getattr(torch, "_clipper_patched_load", False):
            _orig_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load
            torch._clipper_patched_load = True

        self.whisperx_model = whisperx.load_model(
            self.whisperx_model_name, device=self.device, compute_type=compute_type)

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en",
            device=self.device
        )

        print("Transcription models loaded...")

        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            print("Creating gemini client...")
            self.gemini_client = genai.Client(api_key=gemini_key)
            print("Created gemini client...")
        else:
            self.gemini_client = None

    def _ensure_models(self) -> None:
        if self.whisperx_model is None:
            self.load_models()

    def transcribe_video(self, base_dir: pathlib.Path, video_path: pathlib.Path) -> list:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        print("Starting transcription with WhisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device=self.device,
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print("Transcription and alignment took " + str(duration) + " seconds")

        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"],
                })

        return segments

    def identify_moments(self, transcript: list):
        transcript_text = str(transcript)
        max_chars = int(os.getenv("LOCAL_LLM_MAX_CHARS", "4000"))
        if len(transcript_text) > max_chars:
            transcript_text = transcript_text[:max_chars]
        prompt = """
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n""" + transcript_text
        if self.gemini_client is not None:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
            )
            print(f"Identified moments response: ${response.text}")
            return response.text
        local_response = _call_local_llm(prompt)
        print(f"Identified moments response: {local_response}")
        return local_response

    def process_video_action(self, s3_key: str) -> dict:
        self._ensure_models()

        try:
            base_dir, video_path, s3_client = download_video(s3_key)

            # 1. Transcription
            transcript_segments = self.transcribe_video(base_dir, video_path)

            # 2. Identify moments for clips
            print("Identifying clip moments")
            identified_moments_raw = self.identify_moments(transcript_segments)

            clip_moments = parse_clip_moments(identified_moments_raw)
            if not clip_moments:
                clip_moments = _fallback_clip_moments(transcript_segments)

            # 3. Process clips
            for index, moment in enumerate(clip_moments[:5]):
                if "start" in moment and "end" in moment:
                    print("Processing clip" + str(index) + " from " +
                          str(moment["start"]) + " to " + str(moment["end"]))
                    process_clip(base_dir, video_path, s3_key,
                                 moment["start"], moment["end"], index, transcript_segments, s3_client=s3_client)

            return {"status": "completed", "clip_count": len(clip_moments[:5])}
        finally:
            if "base_dir" in locals() and base_dir.exists():
                print(f"Cleaning up temp dir after {base_dir}")
                shutil.rmtree(base_dir, ignore_errors=True)
