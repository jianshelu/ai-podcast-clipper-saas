import os
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from .workflows import HelloWorkflow, ProcessVideoWorkflow
from .activities_cpu import ping
from .activities_video import (
    finalize_activity,
    highlight_activity,
    update_job_activity,
)

async def main():
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "clipper")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE_CPU", "cpu-tq")

    client = await Client.connect(address, namespace=namespace)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[HelloWorkflow, ProcessVideoWorkflow],
        activities=[ping, update_job_activity, highlight_activity, finalize_activity],
    )

    print(f"CPU Worker started. address={address} namespace={namespace} task_queue={task_queue}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
