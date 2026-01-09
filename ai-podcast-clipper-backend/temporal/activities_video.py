# temporal/activities_video.py
from temporalio import activity

from video_processing import VideoProcessor

_processor = None


def _get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
        _processor.load_models()
    return _processor


@activity.defn
def process_video_activity(s3_key: str) -> dict:
    processor = _get_processor()
    return processor.process_video_action(s3_key)
