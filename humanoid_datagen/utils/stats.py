"""Confidence intervals for success rates."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def t_interval(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean over seeds with a Student-t CI (seeds are the unit). Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean()) if v.size else float("nan")
    if v.size < 2:
        return m, float("nan"), float("nan")
    half = sps.t.ppf(0.5 + level / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return m, m - half, m + half


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float, float]:
    """Episode-level Wilson score interval (ignores seed clustering)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = sps.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def cluster_bootstrap(groups: list, level: float = 0.95, n_boot: int = 2000, rng_seed: int = 0):
    """Two-stage bootstrap: resample seeds, then episodes within each seed. groups = list of 0/1 arrays."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    point = float(np.mean([g.mean() for g in groups]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = np.mean([rng.choice(groups[i], len(groups[i])).mean() for i in pick])
    a = (1 - level) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def success_ci(per_seed: dict, method: str, level: float = 0.95, n_boot: int = 2000, rng_seed: int = 0):
    """per_seed: {seed: array of 0/1 successes}. method: t | wilson | bootstrap. Returns dict."""
    arrays = [np.asarray(v, dtype=float) for v in per_seed.values()]
    n_ep = int(sum(a.size for a in arrays))
    if method == "t":
        m, lo, hi = t_interval([a.mean() for a in arrays if a.size], level)
    elif method == "wilson":
        m, lo, hi = wilson_interval(int(sum(a.sum() for a in arrays)), n_ep, level)
    elif method == "bootstrap":
        m, lo, hi = cluster_bootstrap(arrays, level, n_boot, rng_seed)
    else:
        raise ValueError(f"unknown ci method {method!r}")
    return {"mean": m, "lo": lo, "hi": hi, "n_seeds": len(arrays), "n_episodes": n_ep}
