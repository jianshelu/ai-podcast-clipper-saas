import os
from urllib.parse import urlparse
from temporalio import activity

@activity.defn
def process_video_activity(s3_key: str) -> str:
    # 支持 file://
    if s3_key.startswith("file://"):
        local_path = urlparse(s3_key).path
    else:
        local_path = s3_key

    if os.path.exists(local_path):
        activity.logger.info(f"[video] using local file: {local_path}")
        # 下面用 local_path 进入你现有的处理流程（ffmpeg/切片/转码等）
        return f"ok(local): {local_path}"

    # 否则走你原来的 S3 拉取逻辑
    activity.logger.info(f"[video] fetching from s3 key: {s3_key}")
    ...
