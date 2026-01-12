#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path

import requests


def _llm_url(endpoint: str) -> str:
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    return endpoint.rstrip("/") + "/v1/chat/completions"


def run_local_llm(endpoint: str, model: str, prompt: str, timeout: int) -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    response = requests.post(_llm_url(endpoint), headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()

def llm_health_check(endpoint: str, timeout: int) -> None:
    url = endpoint.rstrip("/") + "/v1/models"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        print(f"[OK] local LLM health check: {url}")
    except requests.RequestException as exc:
        print(f"[WARN] local LLM health check failed: {url} err={exc}")

def run(cmd: list[str], *, cwd: str | None = None, env: dict | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

def must_exist(path: Path, desc: str) -> None:
    if not path.exists():
        raise SystemExit(f"[FAIL] missing {desc}: {path}")
    print(f"[OK] {desc}: {path}")

def write_empty_outputs(pywork: Path) -> None:
    pywork.mkdir(parents=True, exist_ok=True)
    empty_tracks = []
    empty_scores = []
    with (pywork / "tracks.pckl").open("wb") as handle:
        pickle.dump(empty_tracks, handle)
    with (pywork / "faces.pckl").open("wb") as handle:
        pickle.dump(empty_tracks, handle)
    with (pywork / "scores.pckl").open("wb") as handle:
        pickle.dump(empty_scores, handle)
    print(f"[WARN] wrote empty whisperVideo outputs into {pywork}")

def _dump_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=True, indent=2, default=str)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video file path")
    ap.add_argument("--workdir", required=True, help="work directory for outputs")
    ap.add_argument("--pywork-dir", help="directory to place tracks/scores outputs")
    ap.add_argument("--whisperx_model", default="large-v2")
    ap.add_argument("--run_whisperx", action="store_true")
    ap.add_argument("--run_whispervideo", action="store_true")
    ap.add_argument("--whispervideo-root", default=os.getenv("WHISPERVIDEO_ROOT", "third_party/whisperVideo"))
    ap.add_argument("--whispervideo-script", default=os.getenv("WHISPERVIDEO_SCRIPT", "inference_folder.py"))
    ap.add_argument("--whispervideo-cpu-fallback", action="store_true")
    ap.add_argument("--run_syncnet", action="store_true")
    ap.add_argument("--run_local_llm", action="store_true")
    ap.add_argument("--llm-endpoint", default=os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8081"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "local-llama"))
    ap.add_argument("--llm-prompt", default=os.getenv("LLM_PROMPT", "Provide tracking hints for the clip."))
    ap.add_argument("--llm-timeout", type=int, default=int(os.getenv("LLM_TIMEOUT", "60")))
    ap.add_argument("--export-json", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    pywork_dir = Path(args.pywork_dir).expanduser().resolve() if args.pywork_dir else None
    if pywork_dir:
        pywork_dir.mkdir(parents=True, exist_ok=True)

    must_exist(video, "input video")

    # 1) Extract audio for WhisperX
    audio_wav = workdir / "audio.wav"
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-ac", "1", "-ar", "16000", "-vn",
        str(audio_wav)
    ])
    must_exist(audio_wav, "audio.wav (16k mono)")

    # 2) WhisperX (CLI)
    # WhisperX 官方示例：直接 `whisperx path/to/audio.wav`，也可加 --model / --diarize 等。:contentReference[oaicite:7]{index=7}
    if args.run_whisperx:
        run([
            "whisperx", str(audio_wav),
            "--model", args.whisperx_model,
            "--output_dir", str(workdir),
        ])
        # 产物文件名取决于 whisperx 的输出策略；通常会有同名的 json/srt/vtt 等
        # 这里做一个“宽松断言”：workdir 里至少应出现 json
        json_files = list(workdir.glob("*.json"))
        if not json_files:
            raise SystemExit(f"[FAIL] WhisperX produced no *.json in {workdir}")
        print(f"[OK] WhisperX json: {json_files[0]}")

    # 3) Columbia/ASD：推荐用 whisperVideo（它会生成 pywork/scores.pckl 等）:contentReference[oaicite:8]{index=8}
    if args.run_whispervideo:
        tp = Path(args.whispervideo_root).expanduser().resolve()
        must_exist(tp, "whisperVideo root")

        # whisperVideo 以“video_folder”为输入概念（README 提到 video_folder/pywork 下会落 scores.pckl 等）:contentReference[oaicite:9]{index=9}
        # 这里创建一个视频目录并把文件放进去（命名不强绑定的话，你也可以只放原始视频）
        video_folder = workdir / "video_folder"
        video_folder.mkdir(exist_ok=True)
        # 复制/软链都行，这里用复制更稳
        vf = video_folder / video.name
        if not vf.exists():
            run(["cp", "-f", str(video), str(vf)])

        # 具体运行脚本请以 whisperVideo 仓库 README 的 quick start 为准（例如 inference_folder*.py）:contentReference[oaicite:10]{index=10}
        # 下面这行是“入口占位”，你需要改成你最终选用的 inference 脚本与参数。
        script_path = Path(args.whispervideo_script)
        if not script_path.is_absolute():
            script_path = tp / script_path
        must_exist(script_path, "whisperVideo script")
        whisper_env = os.environ.copy()
        whisper_env["PYTHONPATH"] = f"{tp}:{whisper_env.get('PYTHONPATH', '')}".rstrip(":")
        pywork = video_folder / "pywork"
        cpu_fallback = args.whispervideo_cpu_fallback or os.getenv("WHISPERVIDEO_CPU_FALLBACK", "").lower() in (
            "1", "true", "yes"
        ) or os.getenv("COLUMBIA_WHISPERVIDEO_CPU_FALLBACK", "").lower() in ("1", "true", "yes")
        whisper_failed = False
        try:
            run(["python", str(script_path), "--videoFolder", str(video_folder)], cwd=str(tp), env=whisper_env)
        except Exception as exc:
            whisper_failed = True
            if cpu_fallback:
                print(f"[WARN] whisperVideo failed: {exc}")
                write_empty_outputs(pywork)
            else:
                raise

        if not whisper_failed:
            # 断言关键输出
            must_exist(pywork / "scores.pckl", "whisperVideo scores.pckl")
            # 他们 README 里叫 faces.pckl（等价于你们说的 tracks）:contentReference[oaicite:11]{index=11}
            must_exist(pywork / "faces.pckl", "whisperVideo faces.pckl (face tracks)")
        if pywork_dir:
            run(["cp", "-f", str(pywork / "scores.pckl"), str(pywork_dir / "scores.pckl")])
            run(["cp", "-f", str(pywork / "faces.pckl"), str(pywork_dir / "faces.pckl")])
        if args.export_json:
            scores = pickle.load((pywork / "scores.pckl").open("rb"))
            faces = pickle.load((pywork / "faces.pckl").open("rb"))
            out_dir = pywork_dir or pywork
            _dump_json(scores, out_dir / "scores.json")
            _dump_json(faces, out_dir / "faces.json")
            tracks_pckl = pywork / "tracks.pckl"
            if tracks_pckl.exists():
                tracks = pickle.load(tracks_pckl.open("rb"))
                _dump_json(tracks, out_dir / "tracks.json")
            print(f"[OK] exported json outputs into {out_dir}")

    # 4) 可选：SyncNet 作为另一条 tracks.pckl 参考来源（能做多脸说话人判断）:contentReference[oaicite:12]{index=12}
    if args.run_syncnet:
        tp = Path("third_party/syncnet_python").resolve()
        must_exist(tp, "third_party/syncnet_python")
        # syncnet_python 的具体运行方式以其 README/demo 为准；
        # 它生态里通常会生成 tracks.pckl 这类文件。:contentReference[oaicite:13]{index=13}
        print("[INFO] SyncNet integration is repo-specific; wire the correct command per its README.")

    if args.run_local_llm:
        llm_health_check(args.llm_endpoint, args.llm_timeout)
        result = run_local_llm(args.llm_endpoint, args.llm_model, args.llm_prompt, args.llm_timeout)
        output_path = (pywork_dir or workdir) / "llm_response.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle)
        print(f"[OK] local LLM response: {output_path}")

    print("[DONE]")

if __name__ == "__main__":
    main()
