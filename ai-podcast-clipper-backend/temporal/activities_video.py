# temporal/activities_video.py
import shutil
import threading

from temporalio import activity

from video_processing import VideoProcessor, download_video, parse_clip_moments, process_clip

_processor = None


def _get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        processor = VideoProcessor()
        processor.load_models()
        _processor = processor
    return _processor


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


def _run_with_heartbeat(func, interval_seconds: int = 30):
    stop_event = threading.Event()

    def _heartbeat_loop():
        while not stop_event.wait(interval_seconds):
            try:
                activity.heartbeat("working")
            except RuntimeError:
                break

    thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    thread.start()
    try:
        return func()
    finally:
        stop_event.set()
        thread.join(timeout=interval_seconds)


@activity.defn
def update_job_activity(s3_key: str, status: str, stage: str) -> dict:
    print(f"[job] s3_key={s3_key} status={status} stage={stage}")
    return {"s3_key": s3_key, "status": status, "stage": stage}


@activity.defn
def transcribe_activity(s3_key: str) -> list:
    processor = _get_processor()
    base_dir = None
    try:
        base_dir, video_path, _ = download_video(s3_key)
        return _run_with_heartbeat(lambda: processor.transcribe_video(base_dir, video_path))
    finally:
        if base_dir and base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


@activity.defn
def highlight_activity(transcript_segments: list) -> list:
    processor = _get_processor()
    identified_moments_raw = processor.identify_moments(transcript_segments)
    clip_moments = parse_clip_moments(identified_moments_raw)
    if not clip_moments:
        print("No clip moments from LLM; using fallback clip window.")
        clip_moments = _fallback_clip_moments(transcript_segments)
    return clip_moments


@activity.defn
def render_clips_activity(s3_key: str, transcript_segments: list, clip_moments: list) -> dict:
    base_dir = None
    try:
        base_dir, video_path, s3_client = download_video(s3_key)
        results = []

        def _render():
            for index, moment in enumerate(clip_moments[:5]):
                if "start" in moment and "end" in moment:
                    print("Processing clip" + str(index) + " from " +
                          str(moment["start"]) + " to " + str(moment["end"]))
                    result = process_clip(
                        base_dir,
                        video_path,
                        s3_key,
                        moment["start"],
                        moment["end"],
                        index,
                        transcript_segments,
                        s3_client=s3_client,
                    )
                    results.append(result)
                    activity.heartbeat({"clip_index": index, "output": result.get("output_s3_key")})

            return results

        rendered = _run_with_heartbeat(_render)
        return {"rendered": rendered, "clip_count": len(rendered)}
    finally:
        if base_dir and base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


@activity.defn
def finalize_activity(s3_key: str, status: str) -> dict:
    print(f"[job] s3_key={s3_key} status={status}")
    return {"s3_key": s3_key, "status": status}
