from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class InjectionSpec:
    """Defines one synthetic anomaly injection on a [T,S,F] tensor."""

    kind: Literal["spike", "level_shift", "dropout", "noise"]
    service_idx: int
    feature_idx: int
    start_t: int
    duration: int
    magnitude: float


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def _validate_bounds(data: np.ndarray, spec: InjectionSpec) -> None:
    if data.ndim != 3:
        raise ValueError(f"Expected data shape [T,S,F], got {data.shape}")
    t, s, f = data.shape
    if not (0 <= spec.service_idx < s):
        raise ValueError(f"service_idx out of range: {spec.service_idx} not in [0,{s})")
    if not (0 <= spec.feature_idx < f):
        raise ValueError(f"feature_idx out of range: {spec.feature_idx} not in [0,{f})")
    if spec.duration <= 0:
        raise ValueError("duration must be > 0")
    if not (0 <= spec.start_t < t):
        raise ValueError(f"start_t out of range: {spec.start_t} not in [0,{t})")


def inject_anomalies(
    data: np.ndarray,
    specs: list[InjectionSpec],
    *,
    clip_range: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies into a copy of data.

    Returns:
        (data_injected, labels)
        labels is a boolean array shape [T], true where any spec affects the timepoint.

    Notes:
        This assumes input features are already normalized (common in this repo).
        If you inject on raw features, consider using more domain-aware magnitudes.
    """

    injected = np.array(data, copy=True)
    t, _, _ = injected.shape
    labels = np.zeros(t, dtype=bool)

    for spec in specs:
        _validate_bounds(injected, spec)
        end_t = min(t, spec.start_t + spec.duration)
        sl = slice(spec.start_t, end_t)

        labels[sl] = True
        series = injected[sl, spec.service_idx, spec.feature_idx]

        if spec.kind == "spike":
            # Multiplicative spike; magnitude=2 means double.
            injected[sl, spec.service_idx, spec.feature_idx] = series * spec.magnitude
        elif spec.kind == "level_shift":
            # Additive shift; magnitude=0.3 shifts upward.
            injected[sl, spec.service_idx, spec.feature_idx] = series + spec.magnitude
        elif spec.kind == "dropout":
            # Simulate missing/zeroed metric.
            injected[sl, spec.service_idx, spec.feature_idx] = 0.0
        elif spec.kind == "noise":
            injected[sl, spec.service_idx, spec.feature_idx] = series + (
                np.random.normal(0.0, spec.magnitude, size=series.shape)
            )
        else:
            raise ValueError(f"Unknown kind: {spec.kind}")

    if clip_range is not None:
        lo, hi = clip_range
        injected = np.clip(injected, lo, hi)

    return injected, labels
