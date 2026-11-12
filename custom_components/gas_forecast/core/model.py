# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""Degree-day regressions of daily gas use on a weighted outdoor temperature (today plus a fading memory)."""
from dataclasses import dataclass, field

import numpy as np

# Residual scale (kWh/day) that turns prior standard deviations into ridge weights
NOISE = 3.0


@dataclass
class Day:
    gas: float | None  # kWh, None when unknown or excluded
    temp: float        # weighted outdoor temperature: today plus a fading memory of earlier days, °C
    age: int = 0       # days before the forecast day


def _wls(X, y, w, prior_mean=None, prior_sd=None):
    """Weighted least squares; a Gaussian prior per coefficient adds ridge rows. Returns the coefficients and the
    objective (weighted squared error plus the prior's penalty)."""
    sw = np.sqrt(w)
    A, b = X * sw[:, None], y * sw
    if prior_mean is not None:
        P = np.diag(NOISE / np.asarray(prior_sd, dtype=float))
        A = np.vstack([A, P])
        b = np.concatenate([b, P @ np.asarray(prior_mean, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = A @ coef - b
    return coef, float(r @ r)


@dataclass
class Model:
    """gas = a + b * d (+ q * d², with `curve`), d = max(0, base - temp); plus a fixed offset (the non-heating gas,
    for heating-only fits)."""

    base: float = 15.0
    fit_base: bool = False          # change-point: choose the base by least squares
    window: int | None = 30         # fit on days up to this age; None = every day given
    half_life: float | None = None  # exponential forgetting, days
    prior: dict = field(default_factory=dict)  # coefficient -> (mean, sd)
    min_days: int = 7
    curve: bool = False             # add q·d²: each degree colder costs a little more than the last
    a: float = 0.0
    b: float = 0.0
    q: float = 0.0
    days: int = 0
    offset: float = 0.0

    def fit(self, days: list[Day]) -> bool:
        days = [d for d in days if d.gas is not None and (self.window is None or d.age <= self.window)]
        if len(days) < self.min_days:
            return False
        y = np.array([d.gas for d in days], dtype=float)
        temp = np.array([d.temp for d in days], dtype=float)
        ages = np.array([d.age for d in days], dtype=float)
        w = 0.5 ** (ages / self.half_life) if self.half_life else np.ones(len(days))
        names = ('a', 'b', 'q') if self.curve else ('a', 'b')
        pm = ps = None
        if self.prior:
            pm = [self.prior.get(n, (0.0, 1e6))[0] for n in names]
            ps = [self.prior.get(n, (0.0, 1e6))[1] for n in names]
        best = None
        for base in (np.arange(10.0, 18.01, 0.5) if self.fit_base else [self.base]):
            d = np.maximum(0.0, base - temp)
            X = np.column_stack([np.ones(len(days)), d, d * d] if self.curve else [np.ones(len(days)), d])
            c, obj = _wls(X, y, w, pm, ps)
            if best is None or obj < best[0]:
                best = (obj, float(base), c)
        _, self.base, c = best
        self.a, self.b, self.days = float(c[0]), float(c[1]), len(days)
        self.q = float(c[2]) if self.curve else 0.0
        return True

    def predict(self, temp: float) -> float:
        d = max(0.0, self.base - temp)
        return max(0.0, self.a + self.b * d + self.q * d * d) + self.offset
