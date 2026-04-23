"""
harnessing
----------
Wave Computing Heterogeneous Scheduler.
Automatically dispatches wave-physics workloads to CPU / GPU / NPU.
"""

__version__ = "1.0.0"
__author__  = "Chur Chin"
__email__   = "tpotaoai@gmail.com"

from .device    import DeviceDetector, DeviceType
from .profiler  import TaskProfiler, TaskType
from .scheduler import WaveScheduler

__all__ = [
    "DeviceDetector",
    "DeviceType",
    "TaskProfiler",
    "TaskType",
    "WaveScheduler",
]
