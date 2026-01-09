import asyncio
import uuid
import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import os

from video_processing import VideoProcessor


class ProcessVideoRequest(BaseModel):
    s3_key: str


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("asd", "/asd", copy=True))

app = modal.App("ai-podcast-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clipper-model-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_model(self):
        self.processor = VideoProcessor()
        self.processor.load_models()

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

        if _temporal_enabled():
            return enqueue_video_workflow(s3_key)

        return self.processor.process_video_action(s3_key)


def _temporal_enabled() -> bool:
    return os.getenv("TEMPORAL_VIDEO_QUEUE_ENABLED", "").lower() in ("1", "true", "yes")


def enqueue_video_workflow(s3_key: str) -> dict:
    async def _enqueue() -> dict:
        from temporalio.client import Client
        from temporal.workflows import ProcessVideoWorkflow

        address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
        namespace = os.getenv("TEMPORAL_NAMESPACE", "clipper")
        task_queue = os.getenv("TEMPORAL_TASK_QUEUE_VIDEO", "tq-video")
        workflow_id = f"video-{uuid.uuid4()}"

        client = await Client.connect(address, namespace=namespace)
        handle = await client.start_workflow(
            ProcessVideoWorkflow.run,
            s3_key,
            id=workflow_id,
            task_queue=task_queue,
        )
        return {
            "status": "queued",
            "workflow_id": handle.id,
            "run_id": handle.run_id,
            "task_queue": task_queue,
        }

    return asyncio.run(_enqueue())


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    url = ai_podcast_clipper.process_video.web_url

    payload = {
        "s3_key": "test2/mi630min.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(url, json=payload,
                             headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)
