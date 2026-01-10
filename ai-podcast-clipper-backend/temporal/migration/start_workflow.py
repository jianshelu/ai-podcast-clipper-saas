import argparse
import asyncio
import os
import uuid

from temporalio.client import Client

from .workflows import ProcessVideoWorkflow


def _get_temporal_config() -> tuple[str, str, str]:
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "process-video-queue")
    return address, namespace, task_queue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger the Temporal workflow that replaces Inngest."
    )
    parser.add_argument("s3_key", help="S3 key for the uploaded video")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    address, namespace, task_queue = _get_temporal_config()

    client = await Client.connect(address, namespace=namespace)
    workflow_id = f"process-video-{uuid.uuid4()}"

    handle = await client.start_workflow(
        ProcessVideoWorkflow.run,
        args.s3_key,
        id=workflow_id,
        task_queue=task_queue,
    )

    print(
        "Started workflow. "
        f"workflow_id={handle.id} run_id={handle.run_id}"
    )


if __name__ == "__main__":
    asyncio.run(main())
