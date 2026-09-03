"""Calibration study of the judge — does the PASS/KILL detector actually work?

Every KILL verdict in this project is only trustworthy if the judge is a
*calibrated detector*: it must almost never PASS noise (low false-positive rate),
its power must rise with the true edge, and it must not be fooled by the three
things that inflate a naive Sharpe — fat tails, serial correlation, and a real
edge that only existed for part of the sample. `test_detection_power.py` proved
the judge *can* PASS and *can* KILL on two hand-picked points; this module maps
the whole response surface.

Design
------
The judge (`run_cpcv_returns`) selects the in-sample-best of N configs per CPCV
split and evaluates it out-of-sample, then deflates the winning Sharpe by N. We
feed it synthetic (n_configs x n_bars) return matrices drawn from a known data-
generating process, so we know the ground truth (edge / no edge) and can measure
how often the judge is right.

Each config is an *independent* draw from the same DGP. This is the adversarial
choice: independence maximises the in-sample selection luck, so it is the hardest
case for the Deflated Sharpe to survive — exactly what we want to stress.

The trade-count gate (`min_trades`) is deliberately satisfied (a trade every
`trade_every` bars) so these studies isolate the *statistical* detector (Sharpe /
DSR / drawdown) rather than the liquidity gate. We report the full-gate PASS rate
AND the DSR detector separately, so the two are never conflated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .pbo import pbo_cscv
from .runner import Thresholds, run_cpcv_returns
from .walk_forward import CPCVConfig

# One trading year of daily bars — matches the project's 1d panels. mu targets an
# annualised Sharpe via mu = S_ann / sqrt(ANN_PERIODS) * vol.
ANN_PERIODS = 365.0


@dataclass
class SynthConfig:
    """Shape of a synthetic experiment (held fixed within a study)."""
    n_configs: int = 6          # = N trials the DSR must deflate
    n_bars: int = 912           # ~2.5y of daily bars, like the real panels
    vol: float = 0.01           # per-bar volatility (target std of every DGP)
    trade_every: int = 3        # a trade every k bars -> satisfies min_trades
    freq: str = "1D"


DEFAULT_CPCV = CPCVConfig(n_groups=10, k_test=2, purge=1, embargo=5)


def mu_for_sharpe(ann_sharpe: float, vol: float, ann_periods: float = ANN_PERIODS) -> float:
    """Per-bar mean that yields a target annualised Sharpe at volatility `vol`."""
    return ann_sharpe / np.sqrt(ann_periods) * vol


# --------------------------------------------------------------------------- #
# Data-generating processes. Each returns a length-n 1-D array with per-bar
# volatility ~= vol (so studies are comparable across DGPs) and mean `mu`.
# --------------------------------------------------------------------------- #

def gen_normal(n: int, rng: np.random.Generator, vol: float, mu: float = 0.0) -> np.ndarray:
    """i.i.d. Gaussian — the clean baseline."""
    return rng.normal(mu, vol, n)


def gen_student_t(n: int, rng: np.random.Generator, vol: float, mu: float = 0.0,
                  df: float = 3.0) -> np.ndarray:
    """Heavy-tailed i.i.d. returns, rescaled to the target per-bar volatility.

    Student-t has variance df/(df-2); we rescale so the sample std matches `vol`,
    isolating *kurtosis* as the only thing that changed. df=3 is very heavy
    (finite variance, infinite 4th moment in the limit df->4)."""
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance")
    raw = rng.standard_t(df, n)
    raw = raw * (vol / np.sqrt(df / (df - 2.0)))
    return mu + raw


def gen_ar1(n: int, rng: np.random.Generator, vol: float, mu: float = 0.0,
            phi: float = 0.3) -> np.ndarray:
    """AR(1) returns with stationary std = vol. Positive phi = momentum in the
    return stream itself, which *inflates* a naive Sharpe by understating the
    variance of the mean — the classic way autocorrelation fools a backtest."""
    eps_sd = vol * np.sqrt(max(1.0 - phi * phi, 1e-12))
    eps = rng.normal(0.0, eps_sd, n)
    x = np.empty(n)
    x[0] = mu + rng.normal(0.0, vol)
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + eps[t]
    return x


def gen_regime_shift(n: int, rng: np.random.Generator, vol: float, mu: float = 0.0,
                     active_frac: float = 0.5) -> np.ndarray:
    """A genuine edge that exists only for the first `active_frac` of the sample,
    then dies (mean -> 0). A calibrated CV judge must NOT extrapolate the good
    era across the whole OOS path."""
    k = int(n * active_frac)
    means = np.concatenate([np.full(k, mu), np.zeros(n - k)])
    return rng.normal(0.0, vol, n) + means


Generator = Callable[..., np.ndarray]


def build_matrix(gen: Generator, cfg: SynthConfig, rng: np.random.Generator,
                 **gen_kwargs) -> np.ndarray:
    """Stack n_configs independent draws from `gen` into an (n_configs, n_bars) matrix."""
    return np.vstack([gen(cfg.n_bars, rng, cfg.vol, **gen_kwargs)
                      for _ in range(cfg.n_configs)])


def apply_cost(R: np.ndarray, cost_per_trade: float, trade_every: int) -> np.ndarray:
    """Charge a per-turnover cost on the bars where a trade happens. Turns a gross
    edge into a net one — the recurring project pattern (gross > 0, net killed)."""
    C = np.zeros_like(R)
    C[:, ::trade_every] = cost_per_trade
    return R - C


def _index(n_bars: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC")


def evaluate(R: np.ndarray, cfg: SynthConfig, thr: Optional[Thresholds] = None,
             cpcv: CPCVConfig = DEFAULT_CPCV, with_pbo: bool = False) -> Dict:
    """Run the judge on one synthetic matrix. Returns the judge report augmented
    with the separate PBO overfitting diagnostic (optional; it is the slow part)."""
    n_configs, n_bars = R.shape
    trades = np.zeros_like(R)
    trades[:, ::cfg.trade_every] = 1
    idx = _index(n_bars, cfg.freq)
    grid = [{"c": i} for i in range(n_configs)]
    rep = run_cpcv_returns(R, trades, idx, grid, cpcv, thr or Thresholds())
    if with_pbo and "error" not in rep:
        rep["pbo"] = pbo_cscv(R)
    return rep


@dataclass
class CellResult:
    """Aggregate of n_seeds judge runs for one (regime, parameter) point."""
    label: str
    params: Dict
    n_seeds: int
    n_valid: int
    pass_rate: float             # full 4-gate PASS rate (the headline)
    dsr_pass_rate: float         # P(DSR >= 0.95) alone — the core detector
    median_dsr: float
    median_oos_sharpe_ann: float
    median_mdd: float
    median_pbo: Optional[float] = None

    def as_dict(self) -> Dict:
        d = {
            "label": self.label, "params": self.params, "n_seeds": self.n_seeds,
            "n_valid": self.n_valid, "pass_rate": round(self.pass_rate, 4),
            "dsr_pass_rate": round(self.dsr_pass_rate, 4),
            "median_dsr": round(self.median_dsr, 4),
            "median_oos_sharpe_ann": round(self.median_oos_sharpe_ann, 4),
            "median_mdd": round(self.median_mdd, 4),
        }
        if self.median_pbo is not None:
            d["median_pbo"] = round(self.median_pbo, 4)
        return d


def run_cell(label: str, params: Dict, matrix_fn: Callable[[np.random.Generator], np.ndarray],
             cfg: SynthConfig, n_seeds: int, seed0: int, thr: Optional[Thresholds] = None,
             cpcv: CPCVConfig = DEFAULT_CPCV, with_pbo: bool = False) -> CellResult:
    """Evaluate one experimental cell over `n_seeds` independent synthetic draws.

    `matrix_fn(rng) -> R` builds the (n_configs x n_bars) matrix for one seed, so
    a caller can compose cost overlays, regime shifts, etc. around the generators.
    """
    reps: List[Dict] = []
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        R = matrix_fn(rng)
        rep = evaluate(R, cfg, thr=thr, cpcv=cpcv, with_pbo=with_pbo)
        if "error" not in rep:
            reps.append(rep)
    n_valid = len(reps)
    if n_valid == 0:
        return CellResult(label, params, n_seeds, 0, 0.0, 0.0, 0.0, 0.0, 0.0, None)

    passes = np.array([r["verdict"] == "PASS" for r in reps])
    dsr = np.array([r["deflated_sharpe_ratio"] for r in reps])
    shr = np.array([r["oos_sharpe_annualized"] for r in reps])
    mdd = np.array([r["oos_max_drawdown"] for r in reps])
    pbo_vals = [r["pbo"] for r in reps if "pbo" in r and not np.isnan(r["pbo"])]
    return CellResult(
        label=label, params=params, n_seeds=n_seeds, n_valid=n_valid,
        pass_rate=float(passes.mean()),
        dsr_pass_rate=float((dsr >= 0.95).mean()),
        median_dsr=float(np.median(dsr)),
        median_oos_sharpe_ann=float(np.median(shr)),
        median_mdd=float(np.median(mdd)),
        median_pbo=float(np.median(pbo_vals)) if pbo_vals else None,
    )


# --------------------------------------------------------------------------- #
# The seven studies. Each returns a list of CellResult. Seeds are disjoint
# across cells (seed0 spacing) so no two cells share a random draw.
# --------------------------------------------------------------------------- #

@dataclass
class StudyGrid:
    """Knobs shared by all studies so the report and the tests can dial the same
    resolution (n_seeds) up for the report and down for CI speed."""
    cfg: SynthConfig = field(default_factory=SynthConfig)
    n_seeds: int = 200
    with_pbo: bool = True


def study_null(g: StudyGrid, seed0: int = 1_000) -> List[CellResult]:
    """False-positive rate on pure noise, as the search widens (N configs)."""
    out = []
    for i, n_cfg in enumerate([1, 6, 20, 50]):
        cfg = SynthConfig(n_configs=n_cfg, n_bars=g.cfg.n_bars, vol=g.cfg.vol,
                          trade_every=g.cfg.trade_every, freq=g.cfg.freq)
        out.append(run_cell(
            "null", {"n_configs": n_cfg, "true_ann_sharpe": 0.0},
            lambda rng, c=cfg: build_matrix(gen_normal, c, rng, mu=0.0),
            cfg, g.n_seeds, seed0 + i * 10_000, with_pbo=g.with_pbo))
    return out


def study_power(g: StudyGrid, seed0: int = 2_000) -> List[CellResult]:
    """True-positive rate vs true annualised Sharpe — the detection curve."""
    out = []
    for i, s_ann in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]):
        cfg = g.cfg
        mu = mu_for_sharpe(s_ann, cfg.vol)
        out.append(run_cell(
            "power", {"true_ann_sharpe": s_ann, "n_configs": cfg.n_configs},
            lambda rng, m=mu, c=cfg: build_matrix(gen_normal, c, rng, mu=m),
            cfg, g.n_seeds, seed0 + i * 10_000, with_pbo=g.with_pbo))
    return out


def study_cost(g: StudyGrid, seed0: int = 3_000) -> List[CellResult]:
    """A passing gross edge (ann Sharpe 3) eaten by rising per-trade cost."""
    out = []
    cfg = g.cfg
    mu = mu_for_sharpe(3.0, cfg.vol)
    for i, cost in enumerate([0.0, 0.0005, 0.001, 0.002, 0.004, 0.008]):
        out.append(run_cell(
            "cost", {"gross_ann_sharpe": 3.0, "cost_per_trade": cost},
            lambda rng, m=mu, c=cost, cf=cfg: apply_cost(
                build_matrix(gen_normal, cf, rng, mu=m), c, cf.trade_every),
            cfg, g.n_seeds, seed0 + i * 10_000, with_pbo=g.with_pbo))
    return out


def study_multiplicity(g: StudyGrid, seed0: int = 4_000) -> List[CellResult]:
    """A modest real edge (ann Sharpe 2) vs a widening search: DSR deflation must
    erode the pass rate as N grows even though the true edge is unchanged."""
    out = []
    mu = mu_for_sharpe(2.0, g.cfg.vol)
    for i, n_cfg in enumerate([1, 6, 20, 50, 100]):
        cfg = SynthConfig(n_configs=n_cfg, n_bars=g.cfg.n_bars, vol=g.cfg.vol,
                          trade_every=g.cfg.trade_every, freq=g.cfg.freq)
        out.append(run_cell(
            "multiplicity", {"true_ann_sharpe": 2.0, "n_configs": n_cfg},
            lambda rng, m=mu, c=cfg: build_matrix(gen_normal, c, rng, mu=m),
            cfg, g.n_seeds, seed0 + i * 10_000, with_pbo=g.with_pbo))
    return out


def study_fat_tails(g: StudyGrid, seed0: int = 5_000) -> List[CellResult]:
    """Heavy tails must NOT manufacture a PASS at zero edge (the DSR/PSR non-
    normality correction), and must NOT destroy power on a real edge."""
    out = []
    cfg = g.cfg
    for i, df in enumerate([3.0, 5.0, 10.0]):
        out.append(run_cell(
            "fat_tails_null", {"df": df, "true_ann_sharpe": 0.0},
            lambda rng, d=df, c=cfg: build_matrix(gen_student_t, c, rng, mu=0.0, df=d),
            cfg, g.n_seeds, seed0 + i * 10_000, with_pbo=g.with_pbo))
    mu = mu_for_sharpe(3.0, cfg.vol)
    for j, df in enumerate([3.0, 5.0, 10.0]):
        out.append(run_cell(
            "fat_tails_edge", {"df": df, "true_ann_sharpe": 3.0},
            lambda rng, d=df, m=mu, c=cfg: build_matrix(gen_student_t, c, rng, mu=m, df=d),
            cfg, g.n_seeds, seed0 + (100 + j) * 10_000, with_pbo=g.with_pbo))
    return out


def study_autocorr(g: StudyGrid, seed0: int = 6_000) -> List[CellResult]:
    """Serial correlation at ZERO true edge. Positive phi inflates a naive Sharpe;
    a well-behaved judge should keep its false-positive rate low. Rising pass rate
    with phi is an honest, documented limitation, not a hidden one."""
    out = []
    cfg = g.cfg
    for i, phi in enumerate([0.0, 0.2, 0.4, 0.6]):
        out.append(run_cell(
            "autocorr", {"phi": phi, "true_ann_sharpe": 0.0},
            lambda rng, p=phi, c=cfg: build_matrix(gen_ar1, c, rng, mu=0.0, phi=p),
            cfg, g.n_seeds, seed0 + i * 10_000, with_pbo=g.with_pbo))
    return out


def study_regime_shift(g: StudyGrid, seed0: int = 7_000) -> List[CellResult]:
    """An edge (ann Sharpe 3 while active) present for only part of the sample.
    Less coverage of the alive era should mean fewer PASSes — the judge should not
    extrapolate a dead edge."""
    out = []
    cfg = g.cfg
    mu = mu_for_sharpe(3.0, cfg.vol)
    for i, frac in enumerate([1.0, 0.75, 0.5, 0.25]):
        out.append(run_cell(
            "regime_shift", {"active_frac": frac, "active_ann_sharpe": 3.0},
            lambda rng, f=frac, m=mu, c=cfg: build_matrix(
                gen_regime_shift, c, rng, mu=m, active_frac=f),
            cfg, g.n_seeds, seed0 + i * 10_000, with_pbo=g.with_pbo))
    return out


def study_sample_size(g: StudyGrid, seed0: int = 8_000) -> List[CellResult]:
    """How much data do you need? PASS rate vs series length at fixed true edges.

    Answers the practitioner's real question — 'is my 1-year backtest long enough
    to detect a Sharpe-2 edge?'. `trade_every=1` keeps the min-trades gate cleared
    at every length (250 bars -> 250 trades > 200), so this isolates the pure
    *statistical* effect of sample size, not the liquidity gate. PBO is off (this
    is about detection, not overfitting)."""
    out = []
    i = 0
    for s_ann in (1.0, 2.0, 3.0):
        for n_bars in (250, 500, 912, 1825):
            cfg = SynthConfig(n_configs=g.cfg.n_configs, n_bars=n_bars, vol=g.cfg.vol,
                              trade_every=1, freq=g.cfg.freq)
            mu = mu_for_sharpe(s_ann, cfg.vol)
            out.append(run_cell(
                "sample_size", {"n_bars": n_bars, "true_ann_sharpe": s_ann},
                lambda rng, m=mu, c=cfg: build_matrix(gen_normal, c, rng, mu=m),
                cfg, g.n_seeds, seed0 + i * 1_000, with_pbo=False))
            i += 1
    return out


ALL_STUDIES: Dict[str, Callable[[StudyGrid, int], List[CellResult]]] = {
    "null": study_null,
    "power": study_power,
    "cost": study_cost,
    "multiplicity": study_multiplicity,
    "fat_tails": study_fat_tails,
    "autocorr": study_autocorr,
    "regime_shift": study_regime_shift,
    "sample_size": study_sample_size,
}


def run_all(g: StudyGrid) -> Dict[str, List[Dict]]:
    """Run every study and return a JSON-serialisable {study: [cell dicts]}."""
    result: Dict[str, List[Dict]] = {}
    for name, fn in ALL_STUDIES.items():
        result[name] = [c.as_dict() for c in fn(g)]  # each fn uses its own default seed0
    return result
