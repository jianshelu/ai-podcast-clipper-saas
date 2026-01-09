import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from temporal.workflows import ProcessVideoWorkflow
from temporal.activities_video import process_video_activity  # 以你实际函数名为准


async def main() -> None:
    address = os.getenv("TEMPORAL_ADDRESS", "127.0.0.1:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "clipper")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "tq-video")

    client = await Client.connect(address, namespace=namespace)

    # 同步 activity 必须提供 executor
    max_workers = int(os.getenv("VIDEO_ACTIVITY_WORKERS", "4"))
    activity_executor = ThreadPoolExecutor(max_workers=max_workers)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ProcessVideoWorkflow],
        activities=[process_video_activity],
        activity_executor=activity_executor,
    )

    print(f"[worker-video] address={address} namespace={namespace} tq={task_queue} workers={max_workers}")
    try:
        await worker.run()
    finally:
        activity_executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
