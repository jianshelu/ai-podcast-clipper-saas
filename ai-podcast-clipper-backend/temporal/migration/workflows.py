from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import invoke_process_video


@workflow.defn
class ProcessVideoWorkflow:
    @workflow.run
    async def run(self, s3_key: str) -> str:
        return await workflow.execute_activity(
            invoke_process_video,
            s3_key,
            start_to_close_timeout=timedelta(minutes=20),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
