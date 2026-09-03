[English](README.md) | Русский

# quant-harness

**Честный бэктест-фреймворк.** Его задача — не «найти прибыльную стратегию», а
сделать так, чтобы вы **не смогли обмануть сами себя**, будто нашли. Он выдаёт
механический вердикт `PASS | KILL`, который нельзя задним числом выдать за успех.

Большинство бэктестов лгут по умолчанию: обучение и оценка на одних данных,
look-ahead на закрытии бара, издержки «потом добавим», и — главное — молчаливый
перебор сотен конфигураций, из которых показывают лучшую. quant-harness
закрывает каждую из этих дыр конструктивно.

```python
import numpy as np, pandas as pd
from harness import run_cpcv_returns, Thresholds, CPCVConfig

# 6 конфигов × 900 баров чистого шума → судья ОБЯЗАН выдать KILL
R = np.random.default_rng(0).normal(0.0, 0.01, (6, 900))
T = np.zeros_like(R); T[:, ::3] = 1                       # оборот каждые 3 бара
idx = pd.date_range("2024-01-01", periods=900, freq="1D", tz="UTC")
rep = run_cpcv_returns(R, T, idx, [{"c": i} for i in range(6)],
                       CPCVConfig(n_groups=10, k_test=2), Thresholds())
print(rep["verdict"])   # -> "KILL"
```

## Для кого

Аллокаторы, prop-деск, маркетплейсы ботов и разработчики AI-агентов, которым нужен
**независимый механический вердикт до того, как капитал выделен** под стратегию —
неважно, придумал её человек или сгенерировал LLM-агент. Направьте его на ряд
доходностей (свой, вендора или контрагента) и получите пред-зарегистрированный
PASS/KILL, который не подкручивали после того, как увидели ответ, — как CLI /
библиотечный вызов или как hosted API, который агент может вызвать напрямую.

## Чем это не является

Не генератор сигналов, не источник альфы, не инвестиционная рекомендация и не
обещание доходности. Он отвечает на один узкий вопрос: *виден ли этот edge на этих
данных после реалистичных издержек и поправки на число перебранных конфигураций,
при пред-зарегистрированных порогах?* KILL означает «здесь не видно», а не «прибыли
нет нигде». PASS — не рекомендация вкладывать капитал.

## Что он гарантирует (гардрейлы)

| Гардрейл | Как реализован | Файл |
|---|---|---|
| **Нет look-ahead** | позиция бара `t` = сигнал `t−1` (`held = target_pos.shift(1)`) | `harness/backtest.py` |
| **Честный OOS** | rolling/anchored walk-forward и Combinatorial Purged CV с purge+embargo; веришь только конкатенации нетронутых test-окон | `harness/walk_forward.py` |
| **Реалистичные издержки** | taker fee + slippage на каждый оборот, funding как периодический денежный поток | `harness/backtest.py` |
| **Поправка на множественность** | Deflated Sharpe дефлирует лучший Sharpe по числу перебранных конфигов N; PBO ловит переобучение самого *перебора* | `harness/deflated_sharpe.py`, `harness/pbo.py` |
| **Механический вердикт** | четыре пред-зарегистрированных порога `Thresholds` → PASS/KILL (`run.py` выходит с кодом 2 при KILL; `qh-audit` — 1 при KILL и 2 при невалидном входе) | `harness/runner.py`, `run.py` |

## Судья измерен, а не «вроде работает»

Любой вердикт стоит ровно столько, сколько стоит калибровка судьи. Поэтому судья
прогнан по всей поверхности PASS/KILL на синтетике с известной истиной
(`harness/calibration.py`, отчёт `reports/CALIBRATION.md`, 200 seed/ячейку):

- **False-positive на шуме (N≥6): 0.0%** — судья не пропускает шум.
- **Порог детекции: PASS≥50% при истинном годовом Sharpe ~2.2, ≥90% при ~3.0** —
  честная планка, которую реальная стратегия должна взять.
- **Издержки** режут gross-edge ровно как в реальных прогонах (net-kills-gross).
- **DSR-дефляция**: тот же истинный edge при росте N с 1 до 100 → PASS 82%→16%.
- **Толстые хвосты** (Student-t, df=3) и **автокорреляция** (φ до 0.6) при нулевом
  edge **не пробивают** FPR — поправка на не-нормальность держит.
- **Смена режима**: edge, живший лишь часть выборки, не экстраполируется.

Полный разбор — `reports/FINDINGS-CALIBRATION.md`. Регрессионные гарды —
`tests/test_calibration.py`, `tests/test_detection_power.py`. Методология целиком
(семь способов, которыми бэктест лжёт, и защита от каждого) — `docs/METHODOLOGY.md`
(`docs/METHODOLOGY.ru.md`).

## Установка

```bash
pip install -e .                 # ядро: numpy, scipy, pandas, requests
pip install -e ".[fast]"         # + pyarrow (parquet-кэш скачанных klines)
pip install -e ".[exchange]"     # + ccxt (доп. загрузчики венью)
pip install -e ".[api]"          # + fastapi, uvicorn, pydantic (hosted verdict API)
pip install -e ".[dev]"          # + pytest, httpx
pip install -e ".[dev,api]"      # всё, что нужно полному набору тестов (см. Быстрый старт)
```

Python ≥ 3.11. Текущая версия пакета: **0.3.0**.

## Быстрый старт

```bash
python examples/quickstart.py    # два демо, БЕЗ скачивания данных
python -m pytest                 # 198 passed с extras dev+api (154 passed + 44 API-теста skipped только с dev)
qh-dsr returns.csv --trials 20   # DSR/PSR из CSV доходностей (консольная утилита)
qh-audit --returns returns.csv --trials 20   # PASS/KILL-вердикт + отчёт (см. ниже)
```

`examples/quickstart.py` показывает оба входа: (A) вердикт судьи KILL на шуме /
PASS на реальном edge; (B) leakage-safe движок на одном инструменте (gross→net,
обороты, просадка).

Реальные данные (бесплатные публичные дампы Binance):

```bash
python run.py --symbol BTCUSDT --interval 1h \
  --months 2024-01,2024-02,2024-03,2024-04,2024-05,2024-06
python run.py --synthetic --n-synth 5000            # offline sanity → KILL
python run_calibration.py --seeds 200               # пересобрать отчёт калибровки
```

## Аудит стратегии (`qh-audit`)

CLI (и Python API), который превращает CSV доходностей по периодам в механический
вердикт PASS/KILL — тот же судья, что и в hosted API, но локально и без сетевых
вызовов:

```bash
qh-audit --returns examples/audit_sample_returns.csv --trials 20 \
  --out audit_report.md --json audit_report.json
```

```
qh-audit: VERDICT KILL  (audit_sample_returns)
  data       : 730 periods x 4 configs, 2.00 years, 365 periods/year
  trials     : 20 (effective)
  OOS Sharpe : +1.066 median path, +0.461 worst path
  max DD     : 0.114   trades: 730
  PSR 0.9335   DSR 0.5885 (N=20)   PBO 0.6865
  checks     : trades_ok=yes oos_sharpe_ok=yes drawdown_ok=yes dsr_ok=NO
  ASSUMPTION: trades not provided; turnover assumed at 1 trade(s) per bar for every config
  report     : audit_report.md      json       : audit_report.json
  statistical report, not investment advice
```

Четыре пред-зарегистрированных гейта (те же откалиброванные пороги, что у судьи
выше): минимум сделок (200), минимальный OOS Sharpe (0.7), максимальная просадка
(0.20), минимальный Deflated Sharpe (0.95). PBO выводится как информационная
диагностика и никогда не меняет вердикт. Поддерживаются издержки (`--costs-bps`),
таблица чувствительности к издержкам 0x/0.5x/1x/2x, своя геометрия CPCV и
переопределение порогов — полный список флагов: `qh-audit --help`.
Коды выхода: `0` PASS, `1` KILL, `2` невалидный вход или незаписываемый
`--out`/`--json`, `3` внутренняя ошибка — падение никогда не выходит с кодом `1`,
поэтому его нельзя прочитать как KILL.

```bash
python examples/audit_quickstart.py   # генерирует examples/audit_sample_returns.csv (~1 с, без сети)
```

Python API: `from harness import audit_returns, render_markdown, AuditInputError`.

## Hosted verdict API

Тонкий FastAPI-сервис вокруг ровно того же судьи (`harness.audit.audit_returns`) —
загрузите матрицу доходностей по HTTP и получите откалиброванный вердикт. Сделан
так, чтобы AI-агент (или человек) мог запросить независимый PASS/KILL, ничего не
устанавливая.

```bash
pip install -e ".[api]"
uvicorn api.app:app --host 0.0.0.0 --port 8000 --no-access-log   # или: make api
# интерактивная документация: http://localhost:8000/docs
```

```bash
docker build -t qh-api:dev .          # или: make docker
docker run --rm -p 8000:8000 --env-file api/.env.example qh-api:dev
```

`GET /healthz` всегда открыт. `POST /v1/verdict` принимает матрицу доходностей (и
опционально сделки, частоту, пороги, геометрию CPCV и издержки) и возвращает
вердикт, проверки по гейтам, метрики (OOS Sharpe, DSR, PBO, просадка), таблицу
чувствительности к издержкам, допущения/предупреждения, короткий Markdown-отчёт и
дисклеймер. Опциональная авторизация по API-ключу (`X-API-Key`) и платёжный гейт
(`QH_PAYMENT_GATE`: `noop` бесплатно / `x402` / `nowpayments` — **платные гейты
это заглушки: они выдают 402-челлендж, но не проверяют оплату**, стартуют только с
`QH_ALLOW_STUB_PAYMENT_GATE=1`; прочитайте
`api/README.md#payments-what-is-real-and-what-is-a-stub`, прежде чем полагаться
на них в проде). Полное описание эндпоинтов, кодов ошибок и переменных окружения:
`api/README.md`.

## Публичный API

```python
from harness import (
    run, run_cpcv, run_cpcv_returns, Thresholds,   # судья / вердикт
    run_backtest, Costs, max_drawdown,             # движок
    WalkForwardConfig, CPCVConfig,                 # сплиттеры
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, pbo_cscv,  # статистика
    build_signal, grid_for, all_families,          # семейства сигналов
    audit_returns, render_markdown, AuditInputError,  # qh-audit
)
```

- `run(df, grid, wf, costs, thr)` — walk-forward судья по одному инструменту.
- `run_cpcv(df, grid, cpcv, costs, thr)` — CPCV-вариант, вердикт по медианному пути.
- `run_cpcv_returns(R, trades, index, grid, cpcv, thr)` — судья по готовой матрице
  доходностей (для портфельных/cross-sectional стратегий).
- `audit_returns(R, ...)` — тот же судья, что за `qh-audit` и hosted API.
- `build_signal(df, params)` — семейства momentum / mean_reversion / breakout / carry.

## Структура

```
quant-harness/
├── pyproject.toml            # пакет + зависимости + console-scripts
├── api/                      # hosted verdict API (FastAPI; в wheel не входит)
├── run.py, run_*.py          # исследовательские раннеры (per-spec прогоны)
├── run_calibration.py        # калибровка судьи → reports/CALIBRATION.{json,md}
├── examples/                 # quickstart.py, audit_quickstart.py — data-free демо
├── harness/
│   ├── __init__.py           # публичный API
│   ├── backtest.py           # движок: анти-look-ahead, costs, funding, max_drawdown
│   ├── walk_forward.py       # leakage-safe WFA + CPCV (purge/embargo)
│   ├── deflated_sharpe.py    # DSR / PSR (Bailey & López de Prado) + qh-dsr CLI
│   ├── pbo.py                # Probability of Backtest Overfitting (CSCV)
│   ├── runner.py             # сборка: WFA/CPCV → OOS → DSR → вердикт
│   ├── audit.py              # qh-audit: CSV на входе, PASS/KILL-отчёт на выходе
│   ├── families.py           # параметрические семейства сигналов (грид = N)
│   ├── calibration.py        # исследование детекционной силы судьи
│   ├── data.py               # загрузчик Binance + синтетика (offline fallback)
│   └── {bybit,hyperliquid,basis,listing,cascade,harvest,paper}.py  # венью/стратегии
├── reports/                  # JSON+MD отчёты и FINDINGS-*
└── tests/                    # pytest (198 passed)
```

## Что этим фреймворком уже проверено

Свод — `../RESEARCH-CONCLUSION.md`. Коротко: 10 пред-зарегистрированных спек × 3
венью (Binance/Bybit/Hyperliquid) — **все KILL** на публичных данных; ни один
проверенный механизм не даёт торгуемого edge net реалистичных издержек.
Единственный найденный положительный денежный поток — funding-harvest на
неэффективной венью, и это **исследовательский результат**, а не живой продукт:
небольшой живой paper-трек (27 тиков за 42 календарных дня) сейчас на −1.5%
(около −13%/год в годовом выражении) — что согласуется с тем, что это carry-сделка
с хвостовым риском и реальным базис-риском, а не бесплатные деньги. Позиция —
`../docs/HARVEST-PATH-A.md`, разбор проекта целиком — `../CASE-STUDY.md`
(`../CASE-STUDY-EN.md`), публичный лендинг — `../site/index.html`. Калибровка
(выше) показывает: это вывод **о проверенных рынках**, а не артефакт судьи —
лучший наблюдённый net 0.88 лежит далеко ниже измеренного порога детекции (~2.2).

## Границы честности (что это и что это НЕ)

- **Это** — детектор edge и анти-самообман. Он говорит «этого не видно на этих
  данных с этими издержками», а не «прибыли нет во вселенной».
- **Это не** — исполнительный движок, live-трейдер или источник альфы.
- Пороги `Thresholds` **пред-зарегистрированы**. Крутить их постфактум ради PASS —
  ровно то переобучение, против которого весь фреймворк; так делать нельзя.
- Калибровка — на синтетике с независимыми конфигами (адверсариальный, худший для
  DSR случай). Реальные гриды имеют коррелированные конфиги → дефляция мягче.

## Связь со скиллом

`deflated_sharpe.py` и `walk_forward.py` — канонические реализации из скилла
`quant-backtest-guardrails`, вложены сюда для автономности. Скилл держит
методологию; harness — рабочая основа под неё.

---

MIT License.
