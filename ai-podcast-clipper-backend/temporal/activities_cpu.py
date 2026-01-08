# temporal/activities_cpu.py
from temporalio import activity

@activity.defn
async def ping() -> str:
    return "pong"
