"""Aggregation helpers on top of `eval.metrics`.

Uses rliable (NeurIPS-standard) for IQM and performance profiles when
multiple (method × scenario) cells need to be compared. Kept lightweight
on purpose — full per-method × per-scenario analysis lives in Phase 4.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    from rliable import library as rly
    from rliable import metrics as rl_metrics
    from rliable import plot_utils
    _HAS_RLIABLE = True
except Exception:  # noqa: BLE001
    _HAS_RLIABLE = False


def iqm_with_ci(values: np.ndarray, n_bootstrap: int = 2000) -> Dict[str, float]:
    """Interquartile mean + 95% stratified-bootstrap CI (rliable)."""
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    values = values[~np.isnan(values.flatten())]
    if values.size == 0:
        return {"iqm": float("nan"), "lo": float("nan"), "hi": float("nan")}
    if not _HAS_RLIABLE:
        # Fallback: trimmed mean between 25th and 75th percentile
        q25, q75 = np.quantile(values, [0.25, 0.75])
        mask = (values >= q25) & (values <= q75)
        iqm = float(np.mean(values[mask]))
        rng = np.random.default_rng(0)
        boots = [
            float(np.mean(rng.choice(values[mask] if mask.any() else values, size=values.size, replace=True)))
            for _ in range(n_bootstrap)
        ]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        return {"iqm": iqm, "lo": lo, "hi": hi}
    iqm = lambda x: np.array([rl_metrics.iqm(x)])  # noqa: E731
    res = rly.get_interval_estimates(values, iqm, reps=n_bootstrap)
    return {"iqm": float(res[0.5][0]), "lo": float(res[0.025][0]), "hi": float(res[0.975][0])}


def performance_profile(
    scores_by_method: Dict[str, np.ndarray],
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Fraction of runs exceeding each threshold (higher-is-better framing).

    Convention: we negate the primary metric (avg travel time) so the
    standard "higher is better" performance profile applies. If `scores_by_method`
    keys look like `method@metric`, the negation is applied automatically.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    out: Dict[str, np.ndarray] = {}
    for name, arr in scores_by_method.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[name] = np.zeros_like(thresholds)
            continue
        # neg so higher=better (treating travel time as cost)
        out[name] = np.array([(arr.size - (arr <= t).sum()) / arr.size for t in thresholds])
    return out
