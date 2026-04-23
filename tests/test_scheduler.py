"""
test_scheduler.py  —  python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from harnessing import WaveScheduler, DeviceDetector, DeviceType, TaskProfiler
from harnessing.profiler import TaskType, SWEEP_THRESHOLD


# ── DeviceDetector ──────────────────────────────────────────────────
def test_detector_has_cpu():
    det = DeviceDetector().scan()
    cpu = det.best(DeviceType.CPU)
    assert cpu is not None
    assert cpu.available

def test_detector_summary_string():
    det = DeviceDetector().scan()
    s = det.summary()
    assert "CPU" in s

def test_detector_available_types_includes_cpu():
    det = DeviceDetector().scan()
    assert DeviceType.CPU in det.available_types()


# ── TaskProfiler ────────────────────────────────────────────────────
def test_profiler_lyapunov_single():
    p = TaskProfiler().profile("lyapunov", {"w0": 1.0, "g1": 0.8})
    assert p.task_type == TaskType.LYAPUNOV_SINGLE
    assert p.preferred_device == DeviceType.CPU

def test_profiler_large_sweep_upgrades_to_gpu():
    p = TaskProfiler().profile("lyapunov", {"n_points": SWEEP_THRESHOLD + 5})
    assert p.preferred_device == DeviceType.GPU

def test_profiler_stdp_batch_goes_to_npu():
    p = TaskProfiler().profile("stdp", {"n_samples": 10000})
    assert p.preferred_device == DeviceType.NPU

def test_profiler_synapse_batch_goes_to_npu():
    p = TaskProfiler().profile("synapse", {"n_samples": 5000})
    assert p.preferred_device == DeviceType.NPU

def test_profiler_bifurcation_stays_cpu():
    p = TaskProfiler().profile("bifurcation", {"n_points": 100})
    assert p.preferred_device == DeviceType.CPU

def test_profiler_polariton_goes_to_gpu():
    p = TaskProfiler().profile("polariton", {})
    assert p.preferred_device == DeviceType.GPU


# ── WaveScheduler ───────────────────────────────────────────────────
def test_scheduler_lyapunov_ok():
    sch = WaveScheduler(verbose=False)
    r = sch.run("lyapunov", {"w0": 1.0, "g1": 0.5, "T": 60})
    assert r["status"] == "ok"
    assert "lambda_max" in r["data"]

def test_scheduler_synapse_ok():
    sch = WaveScheduler(verbose=False)
    r = sch.run("synapse", {"theta": 45.0, "intensity": 5e5})
    assert r["status"] == "ok"
    assert r["data"]["plasticity"] == "neutral"

def test_scheduler_stdp_ok():
    sch = WaveScheduler(verbose=False)
    r = sch.run("stdp", {"n_points": 50})
    assert r["status"] == "ok"
    assert len(r["data"]["dW"]) == 50

def test_scheduler_device_used_in_result():
    sch = WaveScheduler(verbose=False)
    r = sch.run("lyapunov", {"T": 50})
    assert r["device_used"] in ("CPU", "GPU", "NPU")

def test_scheduler_elapsed_positive():
    sch = WaveScheduler(verbose=False)
    r = sch.run("synapse", {})
    assert r["elapsed_ms"] > 0

def test_scheduler_history_recorded():
    sch = WaveScheduler(verbose=False)
    sch.run("synapse", {})
    sch.run("stdp",    {})
    assert len(sch.history()) == 2

def test_scheduler_force_cpu():
    sch = WaveScheduler(verbose=False, force="cpu")
    r = sch.run("lyapunov", {"T": 50})
    assert r["device_used"] == "CPU"

def test_scheduler_unknown_function_raises():
    sch = WaveScheduler(verbose=False)
    with pytest.raises(ValueError):
        sch.run("nonexistent_function", {})

def test_benchmark_returns_cpu():
    sch = WaveScheduler(verbose=False)
    bm = sch.benchmark(T_ode=60)
    assert "CPU" in bm
    assert bm["CPU"]["status"] == "ok"
