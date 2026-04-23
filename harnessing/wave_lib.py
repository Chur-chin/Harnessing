"""
wave_lib.py
-----------
Pure Python / NumPy wave-physics computation kernels.
Used by all executors as the common compute backend.
No hardware-specific code here — just math.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from typing import Dict, Any


# ── Core ODE ─────────────────────────────────────────────────────────
def _rhs(t, y, wE, w0, wp, mu, alpha, g1, g2):
    E, dE, x, dx, p, dp = y
    return [
        dE,
        -wE**2 * E - alpha * E**3 + g1 * x,
        dx,
        -w0**2 * x - mu * (x**2 - 1) * dx + g1 * E + g2 * p,
        dp,
        -wp**2 * p + g2 * x,
    ]


def _lyapunov_single(w0: float, g1: float, T: float = 150.0) -> float:
    """
    Compute maximal Lyapunov exponent for one (w0, g1) pair.
    Used by both CPUExecutor (multiprocessing) and GPUExecutor.
    """
    wE=2.0; wp=1.5; mu=0.8; alpha=0.05; g2=0.2
    y0  = [0.2, 0.0, 0.5, 0.0, 0.1, 0.0]
    y0p = [0.2 + 1e-6, 0.0, 0.5, 0.0, 0.1, 0.0]
    t_e = np.arange(0, T, 0.1)

    args = (wE, w0, wp, mu, alpha, g1, g2)
    s    = solve_ivp(_rhs, (0, T), y0,  t_eval=t_e, args=args, rtol=1e-4, atol=1e-6)
    sp   = solve_ivp(_rhs, (0, T), y0p, t_eval=t_e, args=args, rtol=1e-4, atol=1e-6)

    if s.success and sp.success:
        d = np.sqrt(np.sum((s.y - sp.y)**2, axis=0))
        d = np.maximum(d, 1e-300)
        skip = len(t_e) // 3
        return float(np.mean(np.log(d[skip:] / 1e-6)) / 0.1)
    return 0.0


# ── Wave function implementations ─────────────────────────────────────
def _fn_lyapunov(p: Dict) -> Dict:
    w0 = float(p.get("w0", 1.0)); g1 = float(p.get("g1", 0.8))
    T  = float(p.get("T",  200.0))
    lam = _lyapunov_single(w0, g1, T)
    return {"lambda_max": round(lam, 6), "is_chaotic": lam > 0,
            "params": {"w0": w0, "g1": g1, "T": T}}


def _fn_bifurcation(p: Dict) -> Dict:
    w0 = float(p.get("w0", 1.0)); T = float(p.get("T", 250.0))
    n  = int(p.get("n_points", 60))
    wE=2.0; wp=1.5; mu=0.8; alpha=0.05; g2=0.2
    g1_arr = np.linspace(float(p.get("g1_min",0.05)), float(p.get("g1_max",1.5)), n)
    g_out, x_out = [], []
    for g1 in g1_arr:
        t_e = np.arange(150, T, 0.05)
        args = (wE, w0, wp, mu, alpha, g1, g2)
        sol = solve_ivp(_rhs,(0,T),[0.2,0,0.5,0,0.1,0],t_eval=t_e,args=args,rtol=1e-5,atol=1e-7)
        if not sol.success: continue
        peaks,_ = find_peaks(sol.y[2], height=0, distance=4)
        if len(peaks)==0: continue
        vals = sol.y[2][peaks][-40:]
        g_out.extend([round(g1,5)]*len(vals)); x_out.extend(vals.tolist())
    return {"g1_values": g_out, "x_peaks": [round(v,6) for v in x_out],
            "n_peaks": len(x_out)}


def _fn_synapse(p: Dict) -> Dict:
    theta = float(p.get("theta", 45.0))
    I     = float(p.get("intensity", 5e5))
    th    = theta * np.pi / 180
    wI    = np.cos(th)**2; wII = np.sin(th)**2
    dEF   = 5e-3 * I * (wI - wII)
    plasticity = "LTP" if dEF > 1e-4 else ("LTD" if dEF < -1e-4 else "neutral")
    return {"delta_EF_meV": round(dEF*1000,4), "plasticity": plasticity,
            "wI": round(float(wI),4), "wII": round(float(wII),4)}


def _fn_stdp(p: Dict) -> Dict:
    n  = int(p.get("n_points", 200))
    dt = np.linspace(float(p.get("dt_min",-80)), float(p.get("dt_max",80)), n)
    Ap = float(p.get("A_plus",0.01)); Am = float(p.get("A_minus",0.012))
    tp = float(p.get("tau_plus",20)); tm = float(p.get("tau_minus",20))
    dW = np.where(dt>0, Ap*np.exp(-dt/tp), -Am*np.exp(dt/tm))
    return {"delta_t": dt.tolist(), "dW": [round(v,6) for v in dW.tolist()]}


def _fn_polariton(p: Dict) -> Dict:
    band = str(p.get("band","I")); d_nm = float(p.get("d_nm",10))
    n    = int(p.get("n_points",200))
    omega = np.linspace(762,823,n) if band=="I" else np.linspace(1380,1600,n)
    def eps_dl(w,ei,wTO,wLO,g): return ei*(1+(wLO**2-wTO**2)/(wTO**2-w**2-1j*g*w))
    ep = eps_dl(omega,4.87,760,825,5); ez = eps_dl(omega,2.95,1370,1610,5)
    ratio = np.sqrt(-ep/(ez+1e-30j)); phi = np.arctan(1/(ratio+1e-30))
    q = (np.pi+phi)/(d_nm*1e-9)
    return {"omega_cm": omega.tolist(), "q_re": (np.real(q)*1e-6).tolist(),
            "q_im": (np.imag(q)*1e-6).tolist(), "band": band}


def _fn_hbn_epsilon(p: Dict) -> Dict:
    n = int(p.get("n_points",300))
    w = np.linspace(float(p.get("omega_min",600)), float(p.get("omega_max",1700)), n)
    def eps_dl(w,ei,wTO,wLO,g): return ei*(1+(wLO**2-wTO**2)/(wTO**2-w**2-1j*g*w))
    ep = eps_dl(w,4.87,760,825,5); ez = eps_dl(w,2.95,1370,1610,5)
    return {"omega_cm": w.tolist(), "eps_perp_re": np.real(ep).tolist(),
            "eps_par_re": np.real(ez).tolist()}


# ── Function registry ─────────────────────────────────────────────────
WAVE_FUNCTIONS: Dict[str, Any] = {
    "lyapunov":    _fn_lyapunov,
    "bifurcation": _fn_bifurcation,
    "synapse":     _fn_synapse,
    "stdp":        _fn_stdp,
    "polariton":   _fn_polariton,
    "hbn_epsilon": _fn_hbn_epsilon,
}
