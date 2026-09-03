"""Paper-трекер cash-and-carry funding harvest (спека 0008 / HARVEST-PATH-A).

Симуляция дельта-нейтральных позиций (шорт HL-перп + лонг Binance-спот) на
РЕАЛЬНЫХ котировках обеих ног. В отличие от бэктеста, базис-P&L считается по
фактическим ценам спота И перпа (не по допущению «спот=перп») — это и есть
ценность: измерить настоящий базис-дрейф и исполнение до реального капитала.

Ядро — чистые функции new_state/tick (тестируемы). I/O (котировки, персист) — в
run_paper.py. Состояние — JSON-сериализуемый dict.
"""
from __future__ import annotations

from typing import Dict, List


def new_state(capital: float, notional: float) -> dict:
    """Начальное состояние трекера. notional — фиксированный размер ноги на монету."""
    return {
        "capital": capital,
        "notional": notional,
        "positions": {},          # coin -> {spot, perp, entered}
        "cum_funding": 0.0,
        "cum_basis_pnl": 0.0,
        "cum_costs": 0.0,
        "equity": capital,
        "ticks": [],
    }


def tick(state: dict, prices: Dict[str, Dict[str, float]],
         funding_since: Dict[str, float], target: List[str], costs: Dict[str, float],
         ts: str, basis_alert: float = 0.03) -> dict:
    """Один тик трекера (в идеале — раз в день).

    prices[coin] = {"spot": px, "perp": px} — текущие котировки обеих ног.
    funding_since[coin] — funding, накопленный с прошлого тика (доля).
    target — список монет, которые ДОЛЖНЫ быть в портфеле по правилу harvest.
    Мьютирует и возвращает state (позиции переоцениваются, ребалансируются)."""
    notional = state["notional"]
    per_leg = costs["taker"] + costs["slip"]
    tick_funding = 0.0
    tick_basis = 0.0
    tick_cost = 0.0
    warnings: List[str] = []

    # 1) mark-to-market существующих позиций: базис-P&L + funding с прошлого тика
    for coin, pos in list(state["positions"].items()):
        if coin not in prices:
            warnings.append(f"{coin}: нет котировки на тике — позиция заморожена")
            continue
        spot_ret = prices[coin]["spot"] / pos["spot"] - 1.0
        perp_ret = prices[coin]["perp"] / pos["perp"] - 1.0
        basis = notional * (spot_ret - perp_ret)   # long spot + short perp
        tick_basis += basis
        tick_funding += notional * funding_since.get(coin, 0.0)
        if abs(spot_ret - perp_ret) > basis_alert:
            warnings.append(f"{coin}: базис-разъезд {(spot_ret - perp_ret)*100:.1f}% > "
                            f"{basis_alert*100:.0f}% — риск хеджа")
        pos["spot"] = prices[coin]["spot"]
        pos["perp"] = prices[coin]["perp"]

    # 2) ребаланс к target: выход из лишних, вход в новые (издержки на обе ноги)
    target_set = set(target)
    for coin in list(state["positions"].keys()):
        if coin not in target_set:
            tick_cost += notional * 2 * per_leg
            del state["positions"][coin]
    for coin in target:
        if coin not in state["positions"] and coin in prices:
            tick_cost += notional * 2 * per_leg
            state["positions"][coin] = {"spot": prices[coin]["spot"],
                                        "perp": prices[coin]["perp"], "entered": ts}

    state["cum_funding"] += tick_funding
    state["cum_basis_pnl"] += tick_basis
    state["cum_costs"] += tick_cost
    state["equity"] = (state["capital"] + state["cum_funding"]
                       + state["cum_basis_pnl"] - state["cum_costs"])
    state["ticks"].append({
        "ts": ts, "n_positions": len(state["positions"]),
        "funding": round(tick_funding, 4), "basis_pnl": round(tick_basis, 4),
        "cost": round(tick_cost, 4), "equity": round(state["equity"], 2),
        "warnings": warnings,
    })
    return state
