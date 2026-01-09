import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from .workflows import ProcessVideoWorkflow
from .activities_video import (
    finalize_activity,
    highlight_activity,
    render_clips_activity,
    transcribe_activity,
    update_job_activity,
)


async def main() -> None:
    address = os.getenv("TEMPORAL_ADDRESS", "127.0.0.1:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "clipper")
    cpu_queue = os.getenv("TEMPORAL_TASK_QUEUE_CPU", "cpu-tq")
    gpu_queue = os.getenv("TEMPORAL_TASK_QUEUE_GPU", "gpu-tq")

    client = await Client.connect(address, namespace=namespace)

    max_workers = int(os.getenv("VIDEO_ACTIVITY_WORKERS", "4"))
    activity_executor = ThreadPoolExecutor(max_workers=max_workers)

    cpu_worker = Worker(
        client,
        task_queue=cpu_queue,
        workflows=[ProcessVideoWorkflow],
        activities=[update_job_activity, highlight_activity, finalize_activity],
        activity_executor=activity_executor,
    )

    gpu_worker = Worker(
        client,
        task_queue=gpu_queue,
        workflows=[],
        activities=[transcribe_activity, render_clips_activity],
        activity_executor=activity_executor,
    )

    print(
        f"[worker-video] address={address} namespace={namespace} cpu_tq={cpu_queue} gpu_tq={gpu_queue} workers={max_workers}"
    )
    try:
        await asyncio.gather(cpu_worker.run(), gpu_worker.run())
    finally:
        activity_executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
