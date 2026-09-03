"""Request / response schemas for the verdict API (pydantic v2).

Validation policy: anything the judge cannot evaluate honestly is rejected with
a 422 and a message a human or an agent can act on — ragged matrices, NaN/inf,
too few periods, mismatched ``trades``, unsupported ``freq``.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime
from math import comb
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from harness.audit import DEFAULT_ASSUMED_TRADES_PER_BAR, DEFAULT_CPCV, MAX_TRIALS
from harness.runner import Thresholds
from harness.walk_forward import CPCVConfig

MIN_PERIODS = 100
MAX_CPCV_SPLITS = 120          # C(n_groups, k_test) cap — keeps a call bounded

Matrix = Union[List[List[float]], List[float]]

_FREQ_RE = re.compile(r"^(\d+)?([A-Za-z]+)$")
_UNIT_SECONDS: Dict[str, int] = {
    "s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600,
    "d": 86_400, "day": 86_400, "w": 604_800, "wk": 604_800,
}
# Aliases that name a calendar period (no fixed bar length). Checked before any
# case-folding, so that 'M' can never be read as minutes.
_CALENDAR_UNITS = frozenset({"M", "ME", "MS", "mo", "month", "months", "Q", "q", "quarter",
                             "Y", "y", "yr", "year", "years", "A", "a"})
MAX_FREQ_SECONDS = 366 * 86_400   # coarsest spacing accepted: one bar per year


def parse_freq(freq: str) -> int:
    """Bar spacing in seconds from a ``<n><unit>`` string ('1h', '4h', '1d', '15m', '1w').

    'm' means minutes; 'M' would be a calendar month, which has no fixed length —
    months, quarters and years are rejected explicitly. Upper-case H/D/W/S are
    accepted as their lower-case units. The multiplier is capped at one year so the
    judge's date index can always be built.
    """
    m = _FREQ_RE.match(freq.strip())
    unit = m.group(2) if m else ""
    if unit in _CALENDAR_UNITS or unit.lower() in {"mo", "month", "months"}:
        raise ValueError(f"freq {freq!r}: calendar months, quarters and years are not "
                         "supported (no fixed bar length); use d or w, or resample to "
                         "fixed-length bars")
    key = unit if unit in _UNIT_SECONDS else unit.lower()
    if m is None or key not in _UNIT_SECONDS:
        raise ValueError(f"freq {freq!r} is not supported; use <n><unit> with unit in "
                         "s, m/min, h, d, w (e.g. '1h', '4h', '1d'); 'm' means minutes")
    n = int(m.group(1) or 1)
    if n <= 0:
        raise ValueError("freq multiplier must be positive")
    seconds = n * _UNIT_SECONDS[key]
    if seconds > MAX_FREQ_SECONDS:
        raise ValueError(f"freq {freq!r}: bar spacing must be at most one year")
    return seconds


class ThresholdsIn(BaseModel):
    """Pre-registered PASS gates. Defaults mirror ``harness.runner.Thresholds``."""
    model_config = ConfigDict(extra="forbid")

    min_trades: int = Field(Thresholds.min_trades, ge=0)
    min_oos_sharpe: float = Field(Thresholds.min_oos_sharpe, allow_inf_nan=False)
    max_drawdown: float = Field(Thresholds.max_drawdown, gt=0, le=1, allow_inf_nan=False)
    min_dsr: float = Field(Thresholds.min_dsr, ge=0, le=1, allow_inf_nan=False)

    def to_thresholds(self) -> Thresholds:
        return Thresholds(**self.model_dump())


class CPCVIn(BaseModel):
    """Combinatorial Purged CV layout. Defaults are the calibrated geometry
    (``harness.audit.DEFAULT_CPCV`` = reports/CALIBRATION.md: 10 / 2 / 1 / 5), the same
    ones ``qh-audit`` uses."""
    model_config = ConfigDict(extra="forbid")

    n_groups: int = Field(DEFAULT_CPCV.n_groups, ge=3, le=20)
    k_test: int = Field(DEFAULT_CPCV.k_test, ge=1, le=5)
    purge: int = Field(DEFAULT_CPCV.purge, ge=0, le=1000)
    embargo: int = Field(DEFAULT_CPCV.embargo, ge=0, le=1000)

    @model_validator(mode="after")
    def _bounded(self) -> "CPCVIn":
        if self.k_test >= self.n_groups:
            raise ValueError(f"cpcv.k_test ({self.k_test}) must be < cpcv.n_groups ({self.n_groups})")
        n_splits = comb(self.n_groups, self.k_test)
        if n_splits > MAX_CPCV_SPLITS:
            raise ValueError(f"cpcv: C({self.n_groups}, {self.k_test}) = {n_splits} splits exceeds "
                             f"the cap of {MAX_CPCV_SPLITS}; lower n_groups or k_test")
        return self

    def to_config(self) -> CPCVConfig:
        return CPCVConfig(**self.model_dump())


class VerdictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Matrix = Field(..., description=(
        "Per-bar returns as a fraction (0.01 = 1%). Either a (configs x T) matrix — one row "
        "per configuration you tried — or a single series of length T."))
    trades: Optional[Matrix] = Field(None, description=(
        "Same shape as `returns`: number of position changes on each bar. Omitted = one "
        "trade per bar is assumed (an upper bound; the min_trades gate becomes lenient)."))
    freq: str = Field("1d", description="Bar spacing: '1m', '15m', '1h', '4h', '1d', '1w'.")
    start: Optional[datetime] = Field(None, description="ISO timestamp of the first bar (UTC if naive).")
    n_trials: Optional[int] = Field(None, ge=1, le=MAX_TRIALS, description=(
        "Total number of configurations tried, if more than the rows uploaded; the Deflated "
        f"Sharpe is deflated by this number (at most {MAX_TRIALS})."))
    thresholds: ThresholdsIn = Field(default_factory=lambda: ThresholdsIn())
    cpcv: CPCVIn = Field(default_factory=lambda: CPCVIn())
    costs_bps: Optional[float] = Field(None, ge=0, le=10_000, allow_inf_nan=False, description=(
        "Round-trip cost in basis points charged per unit of `trades` on every bar. Omitted = "
        "returns are treated as already net of costs."))
    assume_trades_per_bar: float = Field(DEFAULT_ASSUMED_TRADES_PER_BAR, ge=0, allow_inf_nan=False,
                                         description=(
        "Turnover to ASSUME on every bar when `trades` is omitted (default 1.0). Ignored when "
        "`trades` is supplied; the response lists it under `assumptions`."))

    _R: np.ndarray = PrivateAttr()
    _trades: Optional[np.ndarray] = PrivateAttr(default=None)
    _freq_seconds: int = PrivateAttr()

    @staticmethod
    def _to_matrix(name: str, value: Matrix) -> np.ndarray:
        try:
            arr = np.asarray(value, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be rectangular: every row needs the same length") from exc
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 1-D series or a 2-D (configs x T) matrix")
        if arr.size == 0:
            raise ValueError(f"{name} is empty")
        bad = np.argwhere(~np.isfinite(arr))
        if bad.size:
            i, j = (int(x) for x in bad[0])
            raise ValueError(f"{name} must be finite (no NaN/inf); first offender at "
                             f"[config {i}, bar {j}]")
        return arr

    @model_validator(mode="after")
    def _check(self) -> "VerdictRequest":
        R = self._to_matrix("returns", self.returns)
        n_cfg, T = R.shape
        if T < MIN_PERIODS:
            raise ValueError(f"returns has {T} periods; at least {MIN_PERIODS} are required for "
                             "a meaningful out-of-sample verdict")
        if self.trades is not None:
            trades = self._to_matrix("trades", self.trades)
            if trades.shape != R.shape:
                raise ValueError(f"trades shape {trades.shape} must equal returns shape {R.shape}")
            if (trades < 0).any():
                raise ValueError("trades must be non-negative")
            self._trades = trades
        if self.n_trials is not None and self.n_trials < n_cfg:
            raise ValueError(f"n_trials ({self.n_trials}) must be >= the number of uploaded "
                             f"configs ({n_cfg})")
        self._freq_seconds = parse_freq(self.freq)
        self._R = R
        return self

    # -- accessors used by the app -------------------------------------------------
    @property
    def n_configs(self) -> int:
        return int(self._R.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self._R.shape[1])

    @property
    def freq_seconds(self) -> int:
        return self._freq_seconds

    def returns_matrix(self) -> np.ndarray:
        return self._R

    def trades_matrix(self) -> Optional[np.ndarray]:
        return self._trades


class Metrics(BaseModel):
    oos_sharpe_annualized: Optional[float]
    worst_path_sharpe_annualized: Optional[float]
    oos_max_drawdown: Optional[float]
    psr_vs_zero: Optional[float]
    deflated_sharpe_ratio: Optional[float]
    pbo: Optional[float]
    n_paths: int
    n_configs_tried: int
    oos_bars: int
    approx_oos_trades: int
    ann_factor: float
    # Annualized OOS Sharpe of every CPCV path (median = `oos_sharpe_annualized`,
    # minimum = `worst_path_sharpe_annualized`); the product plots their distribution.
    path_sharpes_annualized: Optional[List[float]] = None


class CostSensitivityRow(BaseModel):
    """The judge re-run at a multiple of `costs_bps` (0x / 0.5x / 1x / 2x); the 1x row is
    the headline verdict."""
    multiplier: float
    costs_bps: float
    oos_sharpe_annualized: Optional[float]
    worst_path_sharpe_annualized: Optional[float]
    oos_max_drawdown: Optional[float]
    deflated_sharpe_ratio: Optional[float]
    verdict: Literal["PASS", "KILL"]


class VerdictResponse(BaseModel):
    verdict: Literal["PASS", "KILL"]
    checks: Dict[str, bool]
    metrics: Metrics
    thresholds: Dict[str, float]
    cpcv: Dict[str, int]
    assumptions: List[str]
    cost_sensitivity: Optional[List[CostSensitivityRow]] = Field(None, description=(
        "Present only when both `trades` and `costs_bps` were supplied; null otherwise."))
    report_md: str
    disclaimer: str
    version: str


def thresholds_dict(thr: Thresholds) -> Dict[str, float]:
    return asdict(thr)
