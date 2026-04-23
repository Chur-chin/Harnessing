"""
device.py
---------
Hardware detection: CPU cores, GPU (CUDA/CuPy), NPU availability.

DeviceType  — enum of available backends
DeviceDetector — scans the system and reports capabilities

Fallback hierarchy:
    NPU → GPU → CPU   (선호도 순)
    연산 유형에 따라 자동 선택
"""

import os
import platform
import multiprocessing
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional


class DeviceType(Enum):
    CPU = auto()
    GPU = auto()
    NPU = auto()


@dataclass
class DeviceInfo:
    """Snapshot of one hardware device."""
    device_type:  DeviceType
    name:         str
    n_cores:      int   = 1       # CPU logical cores  /  GPU SM count
    memory_gb:    float = 0.0
    available:    bool  = True
    backend:      str   = ""      # "multiprocessing" | "cupy" | "torch_npu" | ...
    extra:        dict  = field(default_factory=dict)

    def __str__(self):
        tag = self.device_type.name
        return (f"[{tag}] {self.name} | "
                f"cores/SMs={self.n_cores} | "
                f"mem={self.memory_gb:.1f} GB | "
                f"backend={self.backend}")


class DeviceDetector:
    """
    Scan the current system for available compute devices.

    Usage
    -----
    det = DeviceDetector()
    det.scan()
    print(det.summary())
    cpu  = det.best(DeviceType.CPU)
    gpu  = det.best(DeviceType.GPU)   # None if unavailable
    npu  = det.best(DeviceType.NPU)   # None if unavailable
    """

    def __init__(self):
        self._devices: List[DeviceInfo] = []
        self._scanned = False

    # ── Scan ─────────────────────────────────────────────────────────
    def scan(self) -> "DeviceDetector":
        self._devices = []
        self._scan_cpu()
        self._scan_gpu()
        self._scan_npu()
        self._scanned = True
        return self

    def _scan_cpu(self):
        n_logical  = multiprocessing.cpu_count()
        try:
            import psutil
            mem_gb = psutil.virtual_memory().total / 1e9
        except ImportError:
            mem_gb = 0.0

        self._devices.append(DeviceInfo(
            device_type = DeviceType.CPU,
            name        = platform.processor() or "CPU",
            n_cores     = n_logical,
            memory_gb   = mem_gb,
            available   = True,
            backend     = "multiprocessing",
        ))

    def _scan_gpu(self):
        # Try CuPy (CUDA)
        try:
            import cupy as cp
            n_dev = cp.cuda.runtime.getDeviceCount()
            for i in range(n_dev):
                with cp.cuda.Device(i):
                    mem = cp.cuda.runtime.memGetInfo()
                    mem_gb = mem[1] / 1e9
                props = cp.cuda.runtime.getDeviceProperties(i)
                self._devices.append(DeviceInfo(
                    device_type = DeviceType.GPU,
                    name        = props["name"].decode(),
                    n_cores     = int(props["multiProcessorCount"]),
                    memory_gb   = mem_gb,
                    available   = True,
                    backend     = "cupy",
                    extra       = {"device_id": i},
                ))
            return
        except Exception:
            pass

        # Try PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    self._devices.append(DeviceInfo(
                        device_type = DeviceType.GPU,
                        name        = p.name,
                        n_cores     = p.multi_processor_count,
                        memory_gb   = p.total_memory / 1e9,
                        available   = True,
                        backend     = "torch_cuda",
                        extra       = {"device_id": i},
                    ))
                return
        except Exception:
            pass

        # No GPU found — register unavailable placeholder
        self._devices.append(DeviceInfo(
            device_type = DeviceType.GPU,
            name        = "No GPU detected",
            available   = False,
            backend     = "none",
        ))

    def _scan_npu(self):
        # Try PyTorch NPU (Ascend / Apple Neural Engine surrogate)
        try:
            import torch
            if hasattr(torch, "npu") and torch.npu.is_available():
                for i in range(torch.npu.device_count()):
                    self._devices.append(DeviceInfo(
                        device_type = DeviceType.NPU,
                        name        = f"Ascend NPU {i}",
                        available   = True,
                        backend     = "torch_npu",
                        extra       = {"device_id": i},
                    ))
                return
        except Exception:
            pass

        # Emulate NPU via numpy batched matrix ops (always available)
        self._devices.append(DeviceInfo(
            device_type = DeviceType.NPU,
            name        = "NPU-emulated (numpy)",
            n_cores     = 1,
            available   = True,
            backend     = "numpy_npu",
            extra       = {"emulated": True},
        ))

    # ── Query ─────────────────────────────────────────────────────────
    def best(self, device_type: DeviceType) -> Optional[DeviceInfo]:
        """Return the best available device of given type, or None."""
        candidates = [d for d in self._devices
                      if d.device_type == device_type and d.available]
        if not candidates:
            return None
        # Prefer highest memory
        return max(candidates, key=lambda d: d.memory_gb)

    def available_types(self) -> List[DeviceType]:
        return list({d.device_type for d in self._devices if d.available})

    def summary(self) -> str:
        self._ensure_scanned()
        lines = ["=== Harnessing Device Report ==="]
        for d in self._devices:
            status = "✓" if d.available else "✗"
            lines.append(f"  {status} {d}")
        return "\n".join(lines)

    def _ensure_scanned(self):
        if not self._scanned:
            self.scan()

    def __repr__(self):
        return f"DeviceDetector(devices={len(self._devices)}, scanned={self._scanned})"
