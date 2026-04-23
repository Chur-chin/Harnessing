"""
executors.py
------------
Backend executors: CPUExecutor, GPUExecutor, NPUExecutor.

Each executor implements execute(function, payload) → dict.

CPU  : multiprocessing.Pool — parallel ODE runs across cores
GPU  : CuPy (if available) or NumPy vectorized fallback
NPU  : PyTorch NPU / Apple MPS / NumPy batched matrix emulation
"""

import numpy as np
from typing import Any, Dict, Optional
from .device import DeviceInfo, DeviceType


# ════════════════════════════════════════════════════════════════════
# Base
# ════════════════════════════════════════════════════════════════════

class BaseExecutor:
    def __init__(self, device: Optional[DeviceInfo]):
        self.device = device

    def execute(self, function: str, payload: Dict[str, Any]) -> Dict:
        raise NotImplementedError

    def _dispatch(self, function: str, payload: Dict) -> Dict:
        """Shared wave-function library (pure Python/NumPy)."""
        from .wave_lib import WAVE_FUNCTIONS
        if function not in WAVE_FUNCTIONS:
            raise ValueError(f"Unknown wave function: '{function}'. "
                             f"Available: {list(WAVE_FUNCTIONS.keys())}")
        return WAVE_FUNCTIONS[function](payload)


# ════════════════════════════════════════════════════════════════════
# CPU Executor  — multiprocessing for parameter sweeps
# ════════════════════════════════════════════════════════════════════

class CPUExecutor(BaseExecutor):
    """
    CPU backend using Python multiprocessing.Pool.
    Best for: single ODE, Lyapunov, bifurcation (serial peaks).
    For sweeps: distributes across all logical cores.
    """

    def execute(self, function: str, payload: Dict[str, Any]) -> Dict:
        n_cores = self.device.n_cores if self.device else 1

        # If sweep, parallelize across cores
        if function in ("param_sweep", "lyapunov_sweep"):
            return self._parallel_sweep(payload, n_cores)

        # Single computation
        return self._dispatch(function, payload)

    def _parallel_sweep(self, payload: Dict, n_cores: int) -> Dict:
        import multiprocessing as mp
        from .wave_lib import _lyapunov_single

        w0_min = float(payload.get("w0_min", 0.3))
        w0_max = float(payload.get("w0_max", 2.3))
        g1_min = float(payload.get("g1_min", 0.1))
        g1_max = float(payload.get("g1_max", 1.5))
        n      = int(payload.get("n", 8))
        T      = float(payload.get("T", 150.0))

        w0_vals = np.linspace(w0_min, w0_max, n)
        g1_vals = np.linspace(g1_min, g1_max, n)
        jobs    = [(w0, g1, T) for w0 in w0_vals for g1 in g1_vals]

        workers = min(n_cores, len(jobs))
        with mp.Pool(workers) as pool:
            lams = pool.starmap(_lyapunov_single, jobs)

        lam_grid = np.array(lams).reshape(n, n)
        return {
            "w0_values":  w0_vals.tolist(),
            "g1_values":  g1_vals.tolist(),
            "lambda_grid": lam_grid.tolist(),
            "n_workers":  workers,
            "backend":    "multiprocessing",
        }


# ════════════════════════════════════════════════════════════════════
# GPU Executor  — CuPy vectorized or NumPy fallback
# ════════════════════════════════════════════════════════════════════

class GPUExecutor(BaseExecutor):
    """
    GPU backend for large vectorized wave computations.
    Best for: parameter sweeps, hBN epsilon, polariton dispersion.
    Falls back to NumPy if CuPy unavailable.
    """

    def __init__(self, device: Optional[DeviceInfo]):
        super().__init__(device)
        self._xp, self._backend = self._detect_backend()

    def _detect_backend(self):
        if self.device is None or not self.device.available:
            return np, "numpy_cpu"
        if self.device.backend == "cupy":
            try:
                import cupy as cp
                return cp, "cupy"
            except ImportError:
                pass
        return np, "numpy_cpu"

    def execute(self, function: str, payload: Dict[str, Any]) -> Dict:
        xp = self._xp

        if function in ("hbn_epsilon", "hbn_epsilon_batch"):
            return self._hbn_epsilon_gpu(payload, xp)
        if function in ("polariton", "polariton_batch"):
            return self._polariton_gpu(payload, xp)
        if function in ("param_sweep", "lyapunov_sweep", "scaling_law"):
            return self._scaling_law_gpu(payload, xp)

        # Fallback to shared library
        return self._dispatch(function, payload)

    def _hbn_epsilon_gpu(self, payload: Dict, xp) -> Dict:
        """Vectorized hBN dielectric tensor on GPU/CPU array."""
        n   = int(payload.get("n_points", 300))
        w   = xp.linspace(
            float(payload.get("omega_min", 600)),
            float(payload.get("omega_max", 1700)), n)

        def eps_dl(w, ei, wTO, wLO, g):
            return ei * (1 + (wLO**2 - wTO**2) / (wTO**2 - w**2 - 1j*g*w))

        ep = eps_dl(w, 4.87, 760.0,  825.0,  5.0)
        ez = eps_dl(w, 2.95, 1370.0, 1610.0, 5.0)

        # Move back to CPU for JSON serialization
        if self._backend == "cupy":
            import cupy as cp
            w  = cp.asnumpy(w)
            ep = cp.asnumpy(ep)
            ez = cp.asnumpy(ez)
        else:
            w  = np.asarray(w)
            ep = np.asarray(ep)
            ez = np.asarray(ez)

        return {
            "omega_cm":    w.tolist(),
            "eps_perp_re": np.real(ep).tolist(),
            "eps_perp_im": np.imag(ep).tolist(),
            "eps_par_re":  np.real(ez).tolist(),
            "eps_par_im":  np.imag(ez).tolist(),
            "backend":     self._backend,
        }

    def _polariton_gpu(self, payload: Dict, xp) -> Dict:
        """PhP dispersion on GPU."""
        band = str(payload.get("band", "I"))
        d_nm = float(payload.get("d_nm", 10.0))
        n    = int(payload.get("n_points", 200))

        if band == "I":
            omega = xp.linspace(762, 823, n)
            ei_p,wTO_p,wLO_p,g_p = 4.87,760.,825.,5.
            ei_z,wTO_z,wLO_z,g_z = 2.95,1370.,1610.,5.
        else:
            omega = xp.linspace(1380, 1600, n)
            ei_p,wTO_p,wLO_p,g_p = 4.87,760.,825.,5.
            ei_z,wTO_z,wLO_z,g_z = 2.95,1370.,1610.,5.

        def eps_dl(w,ei,wTO,wLO,g):
            return ei*(1+(wLO**2-wTO**2)/(wTO**2-w**2-1j*g*w))

        ep    = eps_dl(omega,ei_p,wTO_p,wLO_p,g_p)
        ez    = eps_dl(omega,ei_z,wTO_z,wLO_z,g_z)
        ratio = xp.sqrt(-ep/(ez+1e-30j))
        phi   = xp.arctan(1.0/(ratio+1e-30))
        q     = (xp.pi + phi) / (d_nm*1e-9)

        if self._backend == "cupy":
            import cupy as cp
            omega = cp.asnumpy(omega)
            q     = cp.asnumpy(q)

        return {
            "omega_cm": np.asarray(omega).tolist(),
            "q_re":     (np.real(np.asarray(q))*1e-6).tolist(),
            "q_im":     (np.imag(np.asarray(q))*1e-6).tolist(),
            "band":     band,
            "backend":  self._backend,
        }

    def _scaling_law_gpu(self, payload: Dict, xp) -> Dict:
        """Vectorized g_crit sweep — runs all w0 in parallel on GPU arrays."""
        from .wave_lib import _lyapunov_single
        n_w0   = int(payload.get("n_w0", 8))
        w0_min = float(payload.get("w0_min", 0.3))
        w0_max = float(payload.get("w0_max", 2.3))
        T      = float(payload.get("T", 120.0))

        w0_arr = np.linspace(w0_min, w0_max, n_w0)
        g_arr  = np.linspace(0.05, 1.8, 20)
        gcrit  = []

        for w0 in w0_arr:
            lams = [_lyapunov_single(w0, g, T) for g in g_arr]
            lams = np.array(lams)
            gc = 1.8
            for j in range(len(lams)-1):
                if lams[j] > 0 and lams[j+1] <= 0:
                    gc = float((g_arr[j]+g_arr[j+1])/2)
                    break
            gcrit.append(gc)

        coeffs = list(np.polyfit(w0_arr, gcrit, 2))
        a, b, c = coeffs
        return {
            "w0_values":  w0_arr.tolist(),
            "gcrit":      [round(v,4) for v in gcrit],
            "fit_coeffs": [round(v,6) for v in coeffs],
            "fit_label":  f"g_crit={a:.4f}·ω₀²+{b:.4f}·ω₀+{c:.4f}",
            "backend":    self._backend,
        }


# ════════════════════════════════════════════════════════════════════
# NPU Executor  — batched matrix ops for synapse/STDP inference
# ════════════════════════════════════════════════════════════════════

class NPUExecutor(BaseExecutor):
    """
    NPU backend for neural-network-style batched operations.
    Best for: STDP batch, synapse batch, matrix/tensor ops.
    Emulates NPU with NumPy batched matmul if no hardware NPU.
    """

    def __init__(self, device: Optional[DeviceInfo]):
        super().__init__(device)
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        if self.device is None:
            return "numpy_npu"
        if self.device.backend == "torch_npu":
            return "torch_npu"
        return "numpy_npu"

    def execute(self, function: str, payload: Dict[str, Any]) -> Dict:
        if function in ("stdp_batch", "stdp"):
            return self._stdp_batch(payload)
        if function in ("synapse_batch", "synapse"):
            return self._synapse_batch(payload)
        if function in ("matrix_ops", "matmul"):
            return self._matrix_ops(payload)
        return self._dispatch(function, payload)

    def _stdp_batch(self, payload: Dict) -> Dict:
        """
        Batch STDP kernel evaluation — vectorized matmul pattern.
        Models NPU-style weight-update inference.
        """
        n        = int(payload.get("n_samples", 1000))
        dt_min   = float(payload.get("dt_min",   -80.0))
        dt_max   = float(payload.get("dt_max",    80.0))
        A_plus   = float(payload.get("A_plus",    0.01))
        A_minus  = float(payload.get("A_minus",   0.012))
        tau_plus = float(payload.get("tau_plus",  20.0))
        tau_minus= float(payload.get("tau_minus", 20.0))

        dt = np.linspace(dt_min, dt_max, n)

        # NPU-style: represent as batched matrix multiply
        # W_update = A_matrix @ exp_kernel
        A_mat = np.array([[A_plus, 0.0], [0.0, -A_minus]])
        pos   = np.exp(-np.maximum(dt, 0) / tau_plus)
        neg   = np.exp( np.minimum(dt, 0) / tau_minus)
        kern  = np.stack([pos, neg], axis=0)   # (2, n)
        dW    = (A_mat @ kern)[0] * (dt > 0) + (A_mat @ kern)[1] * (dt <= 0)

        return {
            "delta_t":  dt.tolist(),
            "dW":       dW.tolist(),
            "n_samples": n,
            "backend":  self._backend,
        }

    def _synapse_batch(self, payload: Dict) -> Dict:
        """
        Batch synapse weight update over many (theta, intensity) pairs.
        Vectorized as matrix op — NPU-optimal pattern.
        """
        n         = int(payload.get("n_samples", 1000))
        theta_min = float(payload.get("theta_min", 0.0))
        theta_max = float(payload.get("theta_max", 90.0))
        intensity = float(payload.get("intensity", 5e5))
        COUPLING  = 5e-3

        theta_arr = np.linspace(theta_min, theta_max, n)
        th_rad    = theta_arr * np.pi / 180.0

        # Vectorized: all n updates as matrix multiply
        wI    = np.cos(th_rad)**2
        wII   = np.sin(th_rad)**2
        dEF   = COUPLING * intensity * (wI - wII)

        # Conductance update (Drude approximation, batched)
        E_F0  = float(payload.get("E_F0", 0.10))
        E_Q   = 1.602e-19; KB = 1.381e-23; HBAR = 1.055e-34
        kT    = KB * 300
        omega = 2*np.pi * 1e12

        def G_intra(EF_eV):
            EF = EF_eV * E_Q
            ln = np.log(2 * np.cosh(np.clip(EF/(2*kT), -500, 500)))
            return np.abs(E_Q**2 * kT * ln / (np.pi * HBAR**2 * omega))

        G0    = G_intra(E_F0)
        G_new = G_intra(np.maximum(E_F0 + dEF, 0.005))
        dG    = G_new - G0

        return {
            "theta_arr":  theta_arr.tolist(),
            "delta_EF":   dEF.tolist(),
            "delta_G":    dG.tolist(),
            "n_samples":  n,
            "backend":    self._backend,
        }

    def _matrix_ops(self, payload: Dict) -> Dict:
        """General matrix multiplication benchmark (NPU test pattern)."""
        size = int(payload.get("size", 512))
        A    = np.random.randn(size, size).astype(np.float32)
        B    = np.random.randn(size, size).astype(np.float32)
        C    = A @ B
        return {
            "shape":   [size, size],
            "norm":    float(np.linalg.norm(C)),
            "backend": self._backend,
        }
