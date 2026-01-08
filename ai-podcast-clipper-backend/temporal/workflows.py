# temporal/workflows.py
from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from .activities_cpu import ping

@workflow.defn
class HelloWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        r = await workflow.execute_activity(
            ping,
            start_to_close_timeout=timedelta(seconds=10),
        )
        return f"hello {name}, activity says: {r}"
