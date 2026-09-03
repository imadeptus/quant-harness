# Overnight sweep — 2026-07-17T20:23:53.127171Z

Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT | Families: momentum, mean_reversion, breakout, carry | Period: 24 months from 2024-07 | Runtime: 122.5s

**0 PASS out of 16 cells.** PASS = survived walk-forward + Deflated Sharpe(>=0.95) + trades/DD thresholds. Watch PBO: high (>0.5) means the selection process overfits regardless of DSR.

| Rank | Symbol | Family | Verdict | OOS Sharpe (ann) | Max DD | DSR | PBO | Trades | N cfg |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ETHUSDT | carry | **KILL** | 0.849 | 0.588 | 0.0005 | 0.5516 | 396 | 15 |
| 2 | BNBUSDT | carry | **KILL** | 0.188 | 0.364 | 0.0005 | 0.3016 | 94 | 15 |
| 3 | ETHUSDT | momentum | **KILL** | 0.497 | 0.387 | 0.0004 | 0.2183 | 585 | 84 |
| 4 | SOLUSDT | momentum | **KILL** | -0.526 | 0.681 | 0.0004 | 0.3333 | 871 | 84 |
| 5 | BTCUSDT | carry | **KILL** | 0.2 | 0.308 | 0.0003 | 0.4286 | 370 | 15 |
| 6 | SOLUSDT | carry | **KILL** | -0.888 | 0.628 | 0.0002 | 0.254 | 240 | 15 |
| 7 | BTCUSDT | momentum | **KILL** | -0.408 | 0.435 | 0.0001 | 0.0595 | 539 | 84 |
| 8 | BTCUSDT | mean_reversion | **KILL** | -0.448 | 0.239 | 0.0 | 0.0675 | 256 | 50 |
| 9 | SOLUSDT | mean_reversion | **KILL** | -0.465 | 0.248 | 0.0 | 0.0794 | 236 | 50 |
| 10 | BNBUSDT | momentum | **KILL** | -0.778 | 0.455 | 0.0 | 0.3333 | 622 | 84 |
| 11 | ETHUSDT | mean_reversion | **KILL** | -0.814 | 0.41 | 0.0 | 0.0754 | 467 | 50 |
| 12 | ETHUSDT | breakout | **KILL** | -0.814 | 0.319 | 0.0 | 0.0794 | 969 | 12 |
| 13 | BNBUSDT | mean_reversion | **KILL** | -1.243 | 0.372 | 0.0 | 0.0833 | 371 | 50 |
| 14 | BNBUSDT | breakout | **KILL** | -1.622 | 0.425 | 0.0 | 0.0317 | 764 | 12 |
| 15 | BTCUSDT | breakout | **KILL** | -2.018 | 0.381 | 0.0 | 0.004 | 724 | 12 |
| 16 | SOLUSDT | breakout | **KILL** | -3.612 | 0.781 | 0.0 | 0.5079 | 1129 | 12 |


## How to read this
- **No PASS rows → KILL-1**: honest search found no edge surviving costs+DSR. That is a real, money-saving result, not a failure of the run.
- **A PASS row** is a *candidate*, not a green light: confirm funding was attached (carry), check PBO is low, then paper-trade before any capital.
- Carry rows are untrustworthy if the data log warned funding=0.