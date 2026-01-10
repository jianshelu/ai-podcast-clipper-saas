import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from temporalio.client import Client
from temporalio.worker import Worker

from .workflows import HelloGpuWorkflow
from .activities_gpu import ping_gpu
from .activities_video import render_clips_activity, transcribe_activity


async def main():
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "clipper")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE_GPU", "gpu-tq")
    max_workers = int(os.getenv("GPU_ACTIVITY_WORKERS", "4"))

    client = await Client.connect(address, namespace=namespace)
    activity_executor = ThreadPoolExecutor(max_workers=max_workers)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[HelloGpuWorkflow],
        activities=[ping_gpu, transcribe_activity, render_clips_activity],
        activity_executor=activity_executor,
    )

    print(f"GPU Worker started. address={address} namespace={namespace} task_queue={task_queue}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
