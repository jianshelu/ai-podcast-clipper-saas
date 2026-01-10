# temporal/workflows.py
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import os
    from .activities_cpu import ping
    from .activities_gpu import ping_gpu
    from .activities_video import (
        finalize_activity,
        highlight_activity,
        render_clips_activity,
        transcribe_activity,
        update_job_activity,
    )

CPU_TASK_QUEUE = os.getenv("TEMPORAL_TASK_QUEUE_CPU", "cpu-tq")
GPU_TASK_QUEUE = os.getenv("TEMPORAL_TASK_QUEUE_GPU", "gpu-tq")

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
        retry_policy = RetryPolicy(
            maximum_attempts=3,
            non_retryable_error_types=["ValueError"],
        )

        try:
            await workflow.execute_activity(
                update_job_activity,
                args=[s3_key, "running", "transcribe"],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            transcript_segments = await workflow.execute_activity(
                transcribe_activity,
                args=[s3_key],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=60),
                task_queue=GPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            await workflow.execute_activity(
                update_job_activity,
                args=[s3_key, "running", "highlight"],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            clip_moments = await workflow.execute_activity(
                highlight_activity,
                args=[transcript_segments],
                start_to_close_timeout=timedelta(minutes=5),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            await workflow.execute_activity(
                update_job_activity,
                args=[s3_key, "running", "render"],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            render_result = await workflow.execute_activity(
                render_clips_activity,
                args=[s3_key, transcript_segments, clip_moments],
                start_to_close_timeout=timedelta(minutes=60),
                heartbeat_timeout=timedelta(seconds=60),
                task_queue=GPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            await workflow.execute_activity(
                finalize_activity,
                args=[s3_key, "succeeded"],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            return render_result
        except Exception:
            await workflow.execute_activity(
                finalize_activity,
                args=[s3_key, "failed"],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )
            raise
