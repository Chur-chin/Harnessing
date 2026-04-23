"""
profiler.py
-----------
Task Profiler — classifies incoming wave-physics workloads and
estimates compute cost to guide chip selection.

TaskType     — enum of wave computation categories
TaskProfile  — analysis result for a single job
TaskProfiler — analyses payload → returns TaskProfile
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict


class TaskType(Enum):
    """
    Wave computing task categories.
    Each maps to a preferred hardware backend.
    """
    ODE_SINGLE      = auto()   # single ODE integration        → CPU
    ODE_SWEEP       = auto()   # parameter sweep (many ODEs)   → GPU
    LYAPUNOV_SINGLE = auto()   # one Lyapunov exponent         → CPU
    LYAPUNOV_SWEEP  = auto()   # Lyapunov over 2D param grid   → GPU
    BIFURCATION     = auto()   # bifurcation diagram           → CPU (serial peaks)
    SCALING_LAW     = auto()   # g_crit vs w0 sweep            → GPU
    SYNAPSE_SINGLE  = auto()   # single synaptic update        → CPU
    SYNAPSE_BATCH   = auto()   # batch synapse evaluation      → NPU
    STDP_SINGLE     = auto()   # one STDP kernel               → CPU
    STDP_BATCH      = auto()   # large STDP batch              → NPU
    POLARITON       = auto()   # PhP dispersion curve          → GPU (vectorized)
    HBN_EPSILON     = auto()   # dielectric tensor batch       → GPU
    MATRIX_OPS      = auto()   # general matrix / tensor ops   → NPU
    UNKNOWN         = auto()   # fallback                      → CPU


# ── Preferred chip per task type ─────────────────────────────────────
from .device import DeviceType

TASK_PREFERRED_DEVICE: Dict[TaskType, DeviceType] = {
    TaskType.ODE_SINGLE:      DeviceType.CPU,
    TaskType.ODE_SWEEP:       DeviceType.GPU,
    TaskType.LYAPUNOV_SINGLE: DeviceType.CPU,
    TaskType.LYAPUNOV_SWEEP:  DeviceType.GPU,
    TaskType.BIFURCATION:     DeviceType.CPU,
    TaskType.SCALING_LAW:     DeviceType.GPU,
    TaskType.SYNAPSE_SINGLE:  DeviceType.CPU,
    TaskType.SYNAPSE_BATCH:   DeviceType.NPU,
    TaskType.STDP_SINGLE:     DeviceType.CPU,
    TaskType.STDP_BATCH:      DeviceType.NPU,
    TaskType.POLARITON:       DeviceType.GPU,
    TaskType.HBN_EPSILON:     DeviceType.GPU,
    TaskType.MATRIX_OPS:      DeviceType.NPU,
    TaskType.UNKNOWN:         DeviceType.CPU,
}

# Threshold above which a task is considered "sweep/batch"
SWEEP_THRESHOLD = 16   # number of independent jobs


@dataclass
class TaskProfile:
    """Result of profiling a wave-physics job."""
    function:        str
    task_type:       TaskType
    preferred_device: DeviceType
    n_jobs:          int     = 1       # parallelism estimate
    cost_estimate:   float   = 1.0     # relative compute cost (1 = baseline)
    is_batch:        bool    = False
    notes:           str     = ""


class TaskProfiler:
    """
    Analyses (function_name, payload) pairs and returns a TaskProfile.

    Rules
    -----
    - Function name → base TaskType
    - Payload size params (n_points, n_samples, n_w0) → batch decision
    - T (integration time) → cost estimate
    """

    def profile(self, function: str, payload: Dict[str, Any]) -> TaskProfile:
        task_type = self._classify(function, payload)
        n_jobs    = self._estimate_jobs(function, payload)
        cost      = self._estimate_cost(function, payload)
        is_batch  = n_jobs >= SWEEP_THRESHOLD

        # Upgrade single → sweep if batch size is large
        task_type = self._maybe_upgrade(task_type, is_batch)

        preferred = TASK_PREFERRED_DEVICE[task_type]
        notes     = self._notes(task_type, n_jobs, cost)

        return TaskProfile(
            function         = function,
            task_type        = task_type,
            preferred_device = preferred,
            n_jobs           = n_jobs,
            cost_estimate    = cost,
            is_batch         = is_batch,
            notes            = notes,
        )

    # ── Classification ────────────────────────────────────────────────
    def _classify(self, fn: str, pl: Dict) -> TaskType:
        fn = fn.lower()
        if fn == "lyapunov":
            return TaskType.LYAPUNOV_SINGLE
        if fn in ("bifurcation",):
            return TaskType.BIFURCATION
        if fn == "scaling_law":
            return TaskType.SCALING_LAW
        if fn == "synapse":
            return TaskType.SYNAPSE_SINGLE
        if fn == "stdp":
            return TaskType.STDP_SINGLE
        if fn == "polariton":
            return TaskType.POLARITON
        if fn == "hbn_epsilon":
            return TaskType.HBN_EPSILON
        if fn in ("param_sweep", "lyapunov_sweep"):
            return TaskType.LYAPUNOV_SWEEP
        if fn in ("synapse_batch", "stdp_batch"):
            return TaskType.SYNAPSE_BATCH
        if fn in ("matrix_ops", "matmul", "tensor"):
            return TaskType.MATRIX_OPS
        return TaskType.UNKNOWN

    def _maybe_upgrade(self, tt: TaskType, is_batch: bool) -> TaskType:
        """Upgrade single → sweep/batch variant if payload is large."""
        if not is_batch:
            return tt
        upgrades = {
            TaskType.LYAPUNOV_SINGLE: TaskType.LYAPUNOV_SWEEP,
            TaskType.SYNAPSE_SINGLE:  TaskType.SYNAPSE_BATCH,
            TaskType.STDP_SINGLE:     TaskType.STDP_BATCH,
            TaskType.ODE_SINGLE:      TaskType.ODE_SWEEP,
        }
        return upgrades.get(tt, tt)

    # ── Cost / size estimation ────────────────────────────────────────
    def _estimate_jobs(self, fn: str, pl: Dict) -> int:
        """How many independent parallel jobs does this request spawn?"""
        for key in ("n_points", "n_samples", "n_w0", "n_jobs", "n"):
            if key in pl:
                try:
                    return int(pl[key])
                except (ValueError, TypeError):
                    pass
        # Infer from range params
        n_g1 = pl.get("n_g1",  pl.get("n_points", 1))
        n_w0 = pl.get("n_w0",  1)
        try:
            return int(n_g1) * int(n_w0)
        except (ValueError, TypeError):
            return 1

    def _estimate_cost(self, fn: str, pl: Dict) -> float:
        """Relative cost estimate (1.0 = single fast ODE)."""
        T = float(pl.get("T", 200.0))
        n = self._estimate_jobs(fn, pl)
        base_costs = {
            "lyapunov":   2.0,
            "bifurcation": 3.0,
            "scaling_law": 5.0,
            "synapse":    0.1,
            "stdp":       0.05,
            "polariton":  0.5,
            "hbn_epsilon": 0.3,
        }
        base = base_costs.get(fn.lower(), 1.0)
        return round(base * n * (T / 200.0), 2)

    def _notes(self, tt: TaskType, n_jobs: int, cost: float) -> str:
        dev = TASK_PREFERRED_DEVICE[tt].name
        return (f"TaskType={tt.name} | "
                f"preferred={dev} | "
                f"n_jobs={n_jobs} | "
                f"cost≈{cost}")
