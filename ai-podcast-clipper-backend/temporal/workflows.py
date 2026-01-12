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
    async def run(self, s3_key: str | dict) -> dict:
        retry_policy = RetryPolicy(
            maximum_attempts=3,
            non_retryable_error_types=["ValueError"],
        )

        try:
            skip_transcribe = False
            clip_start = 0.0
            clip_seconds = 60.0
            if isinstance(s3_key, dict):
                payload = s3_key
                s3_key = payload.get("s3_key", "")
                skip_transcribe = bool(payload.get("skip_transcribe", False))
                try:
                    clip_start = float(payload.get("clip_start", clip_start))
                except (TypeError, ValueError):
                    clip_start = 0.0
                try:
                    clip_seconds = float(payload.get("clip_seconds", clip_seconds))
                except (TypeError, ValueError):
                    clip_seconds = 60.0
            clip_end = clip_start + max(0.0, clip_seconds)

            await workflow.execute_activity(
                update_job_activity,
                args=[s3_key, "running", "transcribe"],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            if skip_transcribe:
                transcript_segments = []
                clip_moments = [{"start": clip_start, "end": clip_end}]
            else:
                transcript_segments = await workflow.execute_activity(
                    transcribe_activity,
                    args=[s3_key],
                    start_to_close_timeout=timedelta(minutes=30),
                    task_queue=CPU_TASK_QUEUE,
                    retry_policy=retry_policy,
                )

            await workflow.execute_activity(
                update_job_activity,
                args=[s3_key, "running", "highlight"],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue=CPU_TASK_QUEUE,
                retry_policy=retry_policy,
            )

            if not skip_transcribe:
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
                task_queue=CPU_TASK_QUEUE,
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
