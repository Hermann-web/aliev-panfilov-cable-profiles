"""Profile likelihood for a one-dimensional Aliev-Panfilov cable.

The cable is

    u_t = d u_xx + k u (u - a) (1 - u) - u v
    v_t = (epsilon + mu1 v / (mu2 + u)) (-v - k u (u - a - 1))

with the six scalar values in TRUE, or with d replaced by a cosine field.
The observation is a remote pseudo-electrocardiogram, not the cable state:

    phi(t) = - sum_x du/dx (x, t) * (x - x_e) / r(x)^3 * dx,
    r(x) = sqrt((x - x_e)^2 + h^2).

Noise is one additive Gaussian draw whose standard deviation equals a stated
fraction of the root-mean-square of the noise-free trace. A coefficient is
practically identifiable when its profile rises above 3.841459 on both sides
of the optimum. That value is the 95 percent point of the chi-square
distribution with one degree of freedom.

NumPy is the only dependency. The simplex search is implemented in this file.

    python3 practical_identifiability.py --check
    python3 practical_identifiability.py --scaling
    python3 practical_identifiability.py --sweep

--check recomputes the six scalar profiles and compares the pass list with
practical-identifiability.json. --scaling and --sweep rewrite the field JSON
files beside this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "practical-identifiability.json"

# Scalar preset. These six values are the restricted-parameter experiment.
LENGTH = 100.0
T_END = 100.0
N_X = 401
DT = 0.03
TRUE = {"a": 0.15, "d": 1.0, "epsilon": 0.002, "k": 8.0, "mu1": 0.2, "mu2": 0.3}
NAMES = ["a", "d", "epsilon", "k", "mu1", "mu2"]

# Observation geometry: electrode off the cable axis, beyond its far end.
ELECTRODE_X = LENGTH * 1.25
ELECTRODE_H = LENGTH * 0.25

NOISE_FRACTION = 0.02  # declared noise: 2 per cent of the pseudo-ECG RMS
SEED = 20260805
CHI2_95_1DOF = 3.841459
PROFILE_POINTS = 13
PROFILE_SPAN = 0.60  # scan +/- 60 per cent around the true value


OBS_TIMES = np.arange(0.0, T_END, 0.6)  # fixed physical observation grid
CFL_SAFETY = 0.4  # explicit diffusion needs dt <= dx^2 / (2 d); stay below it


def solve(params: dict[str, float]) -> np.ndarray:
    """Integrate the Aliev-Panfilov cable and return the pseudo-ECG trace.

    The time step adapts to the trial diffusivity. Without that, the explicit
    scheme is unstable for `d` above `dx^2 / (2 dt)`, which for this grid is
    1.04, inside the profiled range. Trial values would then diverge for a
    numerical reason and register as sharply bounded, so the scheme would
    manufacture identifiability it has not measured. The observation grid stays
    fixed in physical time so that traces remain comparable across parameters.
    """
    a, d = params["a"], params["d"]
    eps, k = params["epsilon"], params["k"]
    mu1, mu2 = params["mu1"], params["mu2"]

    x = np.linspace(0.0, LENGTH, N_X)
    dx = float(x[1] - x[0])

    dt = min(DT, CFL_SAFETY * dx * dx / max(d, 1.0e-12))
    n_steps = int(math.ceil(T_END / dt))
    dt = T_END / n_steps

    # Planar wave launched from one end, matching the preset's stimulus extent.
    u = np.zeros(N_X)
    u[x <= 0.3 * LENGTH * 0.1] = 1.0
    v = np.zeros(N_X)

    r = np.sqrt((x - ELECTRODE_X) ** 2 + ELECTRODE_H**2)
    lead = -(x - ELECTRODE_X) / r**3  # the d/dx (1/r) weight, precomputed

    trace = np.empty(OBS_TIMES.size)
    next_obs = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(n_steps + 1):
            now = step * dt
            while next_obs < OBS_TIMES.size and OBS_TIMES[next_obs] <= now:
                grad = np.gradient(u, dx)
                trace[next_obs] = float(np.sum(grad * lead) * dx)
                next_obs += 1
            if next_obs >= OBS_TIMES.size:
                break

            lap = np.empty_like(u)
            lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
            lap[0] = (u[1] - u[0]) / (dx * dx)  # zero-flux ends
            lap[-1] = (u[-2] - u[-1]) / (dx * dx)

            du = d * lap + k * u * (u - a) * (1.0 - u) - u * v
            denom = mu2 + u
            dv = (eps + mu1 * v / denom) * (-v - k * u * (u - a - 1.0))

            u = u + dt * du
            v = v + dt * dv
            if not np.all(np.isfinite(u)):
                return np.full(OBS_TIMES.size, np.nan)

    if not np.all(np.isfinite(trace)):
        return np.full(OBS_TIMES.size, np.nan)
    return trace


def chi_square(params: dict[str, float], observed: np.ndarray, sigma: float) -> float:
    model = solve(params)
    if model.shape != observed.shape or not np.all(np.isfinite(model)):
        return 1.0e12
    return float(np.sum((observed - model) ** 2) / (sigma * sigma))


# ---------------------------------------------------------------------------
# Cosine diffusivity. d(x) = D0 * (1 + sum_j c_j cos(j pi x / L)). The counts
# from this block are practical identifiability for that parameterization.
# They are not a result about an unrestricted diffusivity field.
# ---------------------------------------------------------------------------

D0 = 1.0
FIELD_MODES = [1, 2, 3, 4, 5]
# Decaying alternating structure, extended far enough to profile past the
# point where the pseudo-ECG stops bounding the highest modes.
TRUE_COEFF = [0.20, -0.12, 0.08, -0.06, 0.05, -0.04, 0.035, -0.03]


def diffusivity(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """d(x) expanded in cosine modes about a uniform background."""
    field = np.ones_like(x)
    for j, c in enumerate(coeffs, start=1):
        field = field + c * np.cos(j * np.pi * x / LENGTH)
    return D0 * field


def solve_field(coeffs: np.ndarray) -> np.ndarray:
    """Same model and preset, with a spatially varying diffusivity.

    The diffusion term is discretised in conservative flux form so that a
    varying coefficient is handled consistently rather than by differentiating
    the product.
    """
    a, eps, k = TRUE["a"], TRUE["epsilon"], TRUE["k"]
    mu1, mu2 = TRUE["mu1"], TRUE["mu2"]

    x = np.linspace(0.0, LENGTH, N_X)
    dx = float(x[1] - x[0])
    dfield = diffusivity(x, coeffs)
    if np.any(dfield <= 0.0):
        return np.full(OBS_TIMES.size, np.nan)
    d_half = 0.5 * (dfield[:-1] + dfield[1:])  # d at i+1/2

    dt = min(DT, CFL_SAFETY * dx * dx / float(np.max(dfield)))
    n_steps = int(math.ceil(T_END / dt))
    dt = T_END / n_steps

    u = np.zeros(N_X)
    u[x <= 0.3 * LENGTH * 0.1] = 1.0
    v = np.zeros(N_X)

    r = np.sqrt((x - ELECTRODE_X) ** 2 + ELECTRODE_H**2)
    lead = -(x - ELECTRODE_X) / r**3

    trace = np.empty(OBS_TIMES.size)
    next_obs = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(n_steps + 1):
            now = step * dt
            while next_obs < OBS_TIMES.size and OBS_TIMES[next_obs] <= now:
                trace[next_obs] = float(np.sum(np.gradient(u, dx) * lead) * dx)
                next_obs += 1
            if next_obs >= OBS_TIMES.size:
                break

            flux = d_half * (u[1:] - u[:-1]) / dx  # zero-flux at both ends
            div = np.empty_like(u)
            div[1:-1] = (flux[1:] - flux[:-1]) / dx
            div[0] = flux[0] / dx
            div[-1] = -flux[-1] / dx

            du = div + k * u * (u - a) * (1.0 - u) - u * v
            dv = (eps + mu1 * v / (mu2 + u)) * (-v - k * u * (u - a - 1.0))
            u = u + dt * du
            v = v + dt * dv
            if not np.all(np.isfinite(u)):
                return np.full(OBS_TIMES.size, np.nan)

    if not np.all(np.isfinite(trace)):
        return np.full(OBS_TIMES.size, np.nan)
    return trace


def field_chi_square(coeffs: np.ndarray, observed: np.ndarray, sigma: float) -> float:
    model = solve_field(coeffs)
    if model.shape != observed.shape or not np.all(np.isfinite(model)):
        return 1.0e12
    return float(np.sum((observed - model) ** 2) / (sigma * sigma))


def field_scaling(mode_list=None, noise=None, tag="") -> dict:
    """Count identifiable coefficients as the diffusivity field is refined."""
    mode_list = FIELD_MODES if mode_list is None else mode_list
    noise = NOISE_FRACTION if noise is None else noise
    rows = []
    for n_modes in mode_list:
        truth = np.array(TRUE_COEFF[:n_modes])
        clean = solve_field(truth)
        rms = float(np.sqrt(np.mean(clean**2)))
        sigma = noise * rms
        rng = np.random.default_rng(SEED + n_modes)
        observed = clean + rng.normal(0.0, sigma, size=clean.shape)
        chi2_min = field_chi_square(truth, observed, sigma)

        verdicts = []
        for idx in range(n_modes):
            free = [j for j in range(n_modes) if j != idx]
            grid = np.linspace(
                truth[idx] - 0.6 * abs(truth[idx]) - 0.05,
                truth[idx] + 0.6 * abs(truth[idx]) + 0.05,
                9,
            )
            curve = []
            for fixed in grid:
                def objective(vec: np.ndarray, fixed=fixed, idx=idx) -> float:
                    trial = truth.copy()
                    trial[idx] = fixed
                    for slot, value in zip(free, vec):
                        trial[slot] = value
                    return field_chi_square(trial, observed, sigma)

                if free:
                    start = truth[free]
                    _, best = nelder_mead(
                        objective, start, np.maximum(0.25 * np.abs(start), 0.02), 60
                    )
                else:
                    best = objective(np.array([]))
                curve.append(best - chi2_min)
            centre = len(grid) // 2
            left, right = max(curve[: centre + 1]), max(curve[centre:])
            ident = left > CHI2_95_1DOF and right > CHI2_95_1DOF
            verdicts.append(
                {
                    "mode": idx + 1,
                    "true_value": float(truth[idx]),
                    "max_rise_left": float(left),
                    "max_rise_right": float(right),
                    "identifiable": bool(ident),
                }
            )
        n_ident = sum(1 for v in verdicts if v["identifiable"])
        rows.append(
            {
                "n_modes": n_modes,
                "n_identifiable": n_ident,
                "fraction_identifiable": n_ident / n_modes,
                "coefficients": verdicts,
            }
        )
        weakest = min(min(v["max_rise_left"], v["max_rise_right"]) for v in verdicts)
        rows[-1]["weakest_rise"] = weakest
        print(
            f"{tag}  modes {n_modes} noise {noise:.0%}: {n_ident}/{n_modes} identifiable"
            f"  weakest rise {weakest:8.2f}"
        , flush=True)
    return {
        "background_diffusivity": D0,
        "true_coefficients": TRUE_COEFF,
        "noise_fraction_of_rms": noise,
        "threshold_delta_chi_square": CHI2_95_1DOF,
        "rows": rows,
    }


def nelder_mead(
    objective,
    start: np.ndarray,
    scale: np.ndarray,
    max_iter: int = 90,
) -> tuple[np.ndarray, float]:
    """Minimise `objective` over a free-parameter vector, numpy only."""
    n = len(start)
    simplex = [start.copy()]
    for i in range(n):
        point = start.copy()
        point[i] += scale[i]
        simplex.append(point)
    values = [objective(p) for p in simplex]

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < 1.0e-8 * (abs(values[0]) + 1.0):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_ref = objective(reflected)
        if f_ref < values[0]:
            expanded = centroid + 2.0 * (centroid - simplex[-1])
            f_exp = objective(expanded)
            simplex[-1], values[-1] = (
                (expanded, f_exp) if f_exp < f_ref else (reflected, f_ref)
            )
        elif f_ref < values[-2]:
            simplex[-1], values[-1] = reflected, f_ref
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            f_con = objective(contracted)
            if f_con < values[-1]:
                simplex[-1], values[-1] = contracted, f_con
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    best = int(np.argmin(values))
    return simplex[best], values[best]


def profile(name: str, observed: np.ndarray, sigma: float, chi2_min: float) -> dict:
    """Profile likelihood for one parameter, re-optimising all the others."""
    free = [n for n in NAMES if n != name]
    grid = TRUE[name] * np.linspace(1.0 - PROFILE_SPAN, 1.0 + PROFILE_SPAN, PROFILE_POINTS)

    curve = []
    for fixed in grid:
        def objective(vec: np.ndarray, fixed=fixed) -> float:
            trial = {name: float(fixed)}
            for key, value in zip(free, vec):
                trial[key] = float(value)
            if trial["mu2"] <= 0.0 or trial["d"] <= 0.0 or trial["epsilon"] <= 0.0:
                return 1.0e12
            return chi_square(trial, observed, sigma)

        start = np.array([TRUE[n] for n in free])
        _, best = nelder_mead(objective, start, np.maximum(0.1 * np.abs(start), 1e-4))
        curve.append(best - chi2_min)

    curve = [float(c) for c in curve]
    centre = PROFILE_POINTS // 2
    left = max(curve[: centre + 1])
    right = max(curve[centre:])
    bounded_left = left > CHI2_95_1DOF
    bounded_right = right > CHI2_95_1DOF
    if bounded_left and bounded_right:
        verdict = "identifiable"
    elif bounded_left or bounded_right:
        verdict = "one_sided"
    else:
        verdict = "non_identifiable"
    return {
        "parameter": name,
        "true_value": TRUE[name],
        "grid": [float(g) for g in grid],
        "delta_chi_square": curve,
        "max_rise_left": left,
        "max_rise_right": right,
        "bounded_left": bounded_left,
        "bounded_right": bounded_right,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        print("crossover sweep: mode count at base noise, then noise at 5 modes")
        out = {"mode_axis": None, "noise_axis": []}
        out["mode_axis"] = field_scaling([6, 7], NOISE_FRACTION, "[modes]")
        for nz in (0.05, 0.10, 0.20):
            out["noise_axis"].append(field_scaling([5], nz, "[noise]"))
        path = HERE / "identifiability-crossover.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {path.name}")
        return 0

    if args.scaling:
        print("field refinement: identifiable coefficients as the field grows")
        scaling = field_scaling()
        out = HERE / "identifiability-field-scaling.json"
        out.write_text(json.dumps(scaling, indent=2) + "\n")
        print(f"\nwrote {out.name}")
        return 0

    clean = solve(TRUE)
    rms = float(np.sqrt(np.mean(clean**2)))
    sigma = NOISE_FRACTION * rms
    rng = np.random.default_rng(SEED)
    observed = clean + rng.normal(0.0, sigma, size=clean.shape)
    chi2_min = chi_square(TRUE, observed, sigma)

    profiles = [profile(name, observed, sigma, chi2_min) for name in NAMES]
    identifiable = [p["parameter"] for p in profiles if p["verdict"] == "identifiable"]
    one_sided = [p["parameter"] for p in profiles if p["verdict"] == "one_sided"]
    non_identifiable = [
        p["parameter"] for p in profiles if p["verdict"] == "non_identifiable"
    ]

    result = {
        "model": "Aliev-Panfilov 1D cable, kaba2025numerical preset",
        "observation": "pseudo-ECG at a remote off-axis electrode",
        "n_parameters": len(NAMES),
        "noise_fraction_of_rms": NOISE_FRACTION,
        "sigma": sigma,
        "seed": SEED,
        "threshold_delta_chi_square": CHI2_95_1DOF,
        "chi_square_at_truth": chi2_min,
        "n_observation_samples": int(clean.size),
        "identifiable": identifiable,
        "one_sided": one_sided,
        "non_identifiable": non_identifiable,
        "n_identifiable": len(identifiable),
        "profiles": profiles,
    }

    if args.check:
        if not OUT_JSON.exists():
            print("no stored artifact to check")
            return 1
        stored = json.loads(OUT_JSON.read_text())
        same = stored["n_identifiable"] == result["n_identifiable"] and (
            stored["identifiable"] == result["identifiable"]
        )
        print("check:", "ok" if same else "MISMATCH")
        return 0 if same else 1

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(f"observation samples      {clean.size}")
    print(f"noise sigma              {sigma:.6g}  ({NOISE_FRACTION:.0%} of RMS)")
    print(f"chi-square at truth      {chi2_min:.3f}")
    print()
    for p in profiles:
        print(
            f"{p['parameter']:>8s}  rise L {p['max_rise_left']:10.3f}"
            f"  rise R {p['max_rise_right']:10.3f}   {p['verdict']}"
        )
    print()
    print(f"practically identifiable: {len(identifiable)} of {len(NAMES)}  {identifiable}")
    print(f"one-sided:                {one_sided}")
    print(f"non-identifiable:         {non_identifiable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
