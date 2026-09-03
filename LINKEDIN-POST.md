# LinkedIn / соцсети — короткие тизеры (ведут на полный кейс)

Два варианта под ленту. Роль уже проставлена (AI builder); заполни только `https://github.com/imadeptus/quant-harness`
(первым комментарием). Пост-тизер даёт охват, длинный
кейс (`CASE-STUDY.md` / `CASE-STUDY-EN.md`) — глубину. Ссылку лучше первым комментарием
(LinkedIn душит посты с внешними ссылками в теле).

---

## RU

> Я потратил месяцы, пытаясь найти торговый edge на крипте. Не нашёл ни одного.
> И это моя лучшая инженерная работа за год.
>
> Почти любой может показать бэктест с красивым Sharpe. Проблема в том, что
> обычный бэктест **лжёт по умолчанию** — обучение и оценка на одних данных,
> look-ahead, «издержки потом», молчаливый перебор сотен конфигов с показом лучшего.
>
> Я построил research-харнесс, где каждая такая ложь закрыта конструктивно:
> leakage-safe исполнение, walk-forward / CPCV, Deflated Sharpe с поправкой на число
> попыток, реальные издержки, пред-регистрация гипотез до данных.
>
> 10 пред-зарегистрированных гипотез, 3 биржи — **все отклонены**.
>
> А дальше то, что делают редко: я **измерил сам детектор**. Отклоняющий-всё автомат
> выдал бы тот же поток KILL и был бы бесполезен. Прогнал судью по синтетике с
> известной истиной: ложные срабатывания на шуме — 0%, порог детекции ~2.2 годового
> Sharpe. Значит все KILL — правда о рынках, а не сломанный инструмент.
>
> Мораль, которую я забираю: сегодня любой генерит код и стратегии с AI за минуты —
> дефицитный навык не «сгенерить», а **знать, какому выводу верить**. Честный
> отрицательный результат ценнее подогнанного положительного. Доверять можно только
> тому, что проверил — включая собственный AI.
>
> Разбор, код, калибровка — в кейсе (ссылка в комментарии). Python, CPCV, Deflated
> Sharpe, 76 тестов, CI.
>
> Открыт к ролям AI builder. #AI #LLM #python #dataengineering #backtesting

*(первым комментарием:)* Полный разбор + репозиторий: https://github.com/imadeptus/quant-harness

---

## EN

> I spent months trying to find a trading edge in crypto. I found none.
> And it's the best engineering work I did this year.
>
> Almost anyone can show a backtest with a pretty Sharpe. The catch: a normal
> backtest **lies by default** — training and scoring on the same data, look-ahead,
> "costs later," silently trying hundreds of configs and showing the best.
>
> I built a research harness where each of those lies is closed by construction:
> leakage-safe execution, walk-forward / CPCV, Deflated Sharpe corrected for the
> number of trials, realistic costs, hypotheses pre-registered before touching data.
>
> 10 pre-registered hypotheses, 3 exchanges — **all rejected**.
>
> Then the part almost nobody does: I **measured the detector itself**. A
> reject-everything machine would produce the same stream of KILLs and be useless.
> I ran the judge over synthetic data with known ground truth: 0% false positives on
> noise, detection threshold ~2.2 annualized Sharpe. So the KILLs are the truth about
> the markets — not a broken tool.
>
> The lesson I keep: today anyone generates code and strategies with AI in minutes —
> the scarce skill isn't generating, it's **knowing which output to trust**. An honest
> negative beats a fitted positive. You can only trust what you've verified — including
> your own AI.
>
> Write-up, code, calibration in the case study (link in comments). Python, CPCV,
> Deflated Sharpe, 76 tests, CI.
>
> Open to AI builder roles. #AI #LLM #python #dataengineering #backtesting

*(as the first comment:)* Full write-up + repo: https://github.com/imadeptus/quant-harness
