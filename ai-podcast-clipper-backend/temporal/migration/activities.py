import os
import requests
from temporalio import activity


def _get_endpoint_config() -> tuple[str, str]:
    endpoint = os.getenv("PROCESS_VIDEO_ENDPOINT")
    auth_token = os.getenv("PROCESS_VIDEO_ENDPOINT_AUTH")
    if not endpoint or not auth_token:
        raise RuntimeError(
            "PROCESS_VIDEO_ENDPOINT and PROCESS_VIDEO_ENDPOINT_AUTH must be set"
        )
    return endpoint, auth_token


@activity.defn
async def invoke_process_video(s3_key: str) -> str:
    endpoint, auth_token = _get_endpoint_config()

    response = requests.post(
        endpoint,
        json={"s3_key": s3_key},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.text
