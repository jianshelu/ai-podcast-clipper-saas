# temporal/workflows.py
from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from .activities_cpu import ping
    from .activities_gpu import ping_gpu
    from .activities_video import process_video_activity

@workflow.defn
class HelloWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        r = await workflow.execute_activity(
            ping,
            start_to_close_timeout=timedelta(seconds=10),
        )
        return f"hello {name}, activity says: {r}"


@workflow.defn
class HelloGpuWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        r = await workflow.execute_activity(
            ping_gpu,
            start_to_close_timeout=timedelta(seconds=10),
            task_queue="gpu-tq",
        )
        return f"hello {name}, gpu activity says: {r}"


@workflow.defn
class ProcessVideoWorkflow:
    @workflow.run
    async def run(self, s3_key: str) -> dict:
        result = await workflow.execute_activity(
            process_video_activity,
            s3_key,
            start_to_close_timeout=timedelta(minutes=60),
        )
        return result
