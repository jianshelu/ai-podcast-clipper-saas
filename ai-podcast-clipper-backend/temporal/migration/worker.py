import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from .activities import invoke_process_video
from .workflows import ProcessVideoWorkflow


def _get_temporal_config() -> tuple[str, str, str]:
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "process-video-queue")
    return address, namespace, task_queue


async def main() -> None:
    address, namespace, task_queue = _get_temporal_config()
    client = await Client.connect(address, namespace=namespace)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ProcessVideoWorkflow],
        activities=[invoke_process_video],
    )

    print(
        "Temporal worker started. "
        f"address={address} namespace={namespace} task_queue={task_queue}"
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
