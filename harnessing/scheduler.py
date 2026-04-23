"""
scheduler.py
------------
WaveScheduler — the core heterogeneous dispatch engine.

Decision flow
-------------
1. TaskProfiler  → classify workload, estimate size/cost
2. DeviceDetector → know what hardware is available
3. Cost function  → pick optimal chip
4. Executor       → run on chosen backend
5. Fallback       → degrade gracefully if preferred chip unavailable

Fallback chain:  NPU → GPU → CPU  (always succeeds)
"""

import time
from typing import Any, Dict, Optional

from .device    import DeviceDetector, DeviceType, DeviceInfo
from .profiler  import TaskProfiler, TaskProfile, TaskType
from .executors import CPUExecutor, GPUExecutor, NPUExecutor


class WaveScheduler:
    """
    Top-level scheduler: accepts a (function, payload) pair,
    decides which chip to use, executes, and returns the result.

    Parameters
    ----------
    verbose : bool — print scheduling decisions (default True)
    force   : str  — force a specific backend: "cpu" | "gpu" | "npu" | None

    Usage
    -----
    sch = WaveScheduler()
    result = sch.run("lyapunov",    {"w0": 1.0, "g1": 0.8})
    result = sch.run("param_sweep", {"w0_range": [0.3, 2.3], "n": 20})
    result = sch.run("stdp_batch",  {"n_samples": 50000})
    """

    def __init__(self, verbose: bool = True, force: Optional[str] = None):
        self.verbose  = verbose
        self.force    = force.upper() if force else None
        self.detector = DeviceDetector().scan()
        self.profiler = TaskProfiler()
        self._executors = {
            DeviceType.CPU: CPUExecutor(self.detector.best(DeviceType.CPU)),
            DeviceType.GPU: GPUExecutor(self.detector.best(DeviceType.GPU)),
            DeviceType.NPU: NPUExecutor(self.detector.best(DeviceType.NPU)),
        }
        self._history = []   # [(function, device_used, elapsed_ms)]

    # ── Public API ────────────────────────────────────────────────────
    def run(self, function: str, payload: Dict[str, Any] = {}) -> Dict:
        """
        Schedule and execute a wave-physics computation.

        Parameters
        ----------
        function : str   — wave function name
        payload  : dict  — input parameters

        Returns
        -------
        dict with keys:
            status        : "ok" | "error"
            function      : function name
            device_used   : "CPU" | "GPU" | "NPU"
            data          : result payload
            elapsed_ms    : wall-clock time [ms]
            schedule_info : TaskProfile notes
        """
        profile  = self.profiler.profile(function, payload)
        device   = self._select_device(profile)
        executor = self._executors[device]

        if self.verbose:
            self._log(function, profile, device)

        t0 = time.perf_counter()
        try:
            data = executor.execute(function, payload)
            status = "ok"
            error  = None
        except Exception as e:
            # Fallback to CPU
            if device != DeviceType.CPU:
                if self.verbose:
                    print(f"  [Harnessing] ⚠ {device.name} failed → falling back to CPU")
                data   = self._executors[DeviceType.CPU].execute(function, payload)
                device = DeviceType.CPU
            else:
                raise
            status = "ok"
            error  = None

        elapsed = round((time.perf_counter() - t0) * 1000, 3)
        self._history.append((function, device.name, elapsed))

        return {
            "status":        status,
            "function":      function,
            "device_used":   device.name,
            "data":          data,
            "elapsed_ms":    elapsed,
            "schedule_info": profile.notes,
        }

    def device_info(self) -> str:
        """Human-readable hardware summary."""
        return self.detector.summary()

    def history(self):
        """Return dispatch history list."""
        return list(self._history)

    def benchmark(self, T_ode: float = 100.0) -> Dict:
        """
        Quick benchmark: run each executor on a standard task
        and report wall-clock times.
        """
        results = {}
        payload = {"w0": 1.0, "g1": 0.8, "T": T_ode}
        for dtype, executor in self._executors.items():
            dev = self.detector.best(dtype)
            if dev is None or not dev.available:
                results[dtype.name] = {"status": "unavailable"}
                continue
            t0 = time.perf_counter()
            try:
                executor.execute("lyapunov", payload)
                elapsed = round((time.perf_counter() - t0) * 1000, 2)
                results[dtype.name] = {"status": "ok", "elapsed_ms": elapsed}
            except Exception as e:
                results[dtype.name] = {"status": "error", "error": str(e)}
        return results

    # ── Device selection logic ─────────────────────────────────────────
    def _select_device(self, profile: TaskProfile) -> DeviceType:
        """
        Select the best available device for this task.

        Priority:
        1. force override (if set)
        2. preferred device from profile (if available)
        3. fallback chain: NPU → GPU → CPU
        """
        if self.force:
            forced = DeviceType[self.force]
            dev = self.detector.best(forced)
            if dev and dev.available:
                return forced
            if self.verbose:
                print(f"  [Harnessing] forced {self.force} unavailable → CPU")
            return DeviceType.CPU

        preferred = profile.preferred_device
        dev = self.detector.best(preferred)
        if dev and dev.available:
            return preferred

        # Fallback chain
        fallback = [DeviceType.GPU, DeviceType.CPU]
        for dt in fallback:
            d = self.detector.best(dt)
            if d and d.available:
                if self.verbose:
                    print(f"  [Harnessing] {preferred.name} unavailable "
                          f"→ falling back to {dt.name}")
                return dt

        return DeviceType.CPU

    def _log(self, function: str, profile: TaskProfile, device: DeviceType):
        icon = {"CPU": "🔵", "GPU": "🟢", "NPU": "🟣"}.get(device.name, "⚪")
        print(f"  {icon} [{device.name}] {function} "
              f"| task={profile.task_type.name} "
              f"| jobs={profile.n_jobs} "
              f"| cost≈{profile.cost_estimate}")
