# temporal/activities_gpu.py
from temporalio import activity


@activity.defn
async def ping_gpu() -> str:
    return "pong-gpu"
