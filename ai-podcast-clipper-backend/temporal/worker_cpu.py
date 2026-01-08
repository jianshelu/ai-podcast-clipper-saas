import os
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from .workflows import HelloWorkflow
from .activities_cpu import ping

async def main():
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "clipper")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE_CPU", "tq-cpu")

    client = await Client.connect(address, namespace=namespace)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[HelloWorkflow],
        activities=[ping],
    )

    print(f"CPU Worker started. address={address} namespace={namespace} task_queue={task_queue}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
