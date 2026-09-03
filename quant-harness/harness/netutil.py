"""Retrying JSON-over-HTTP helpers for the public exchange endpoints.

Why this exists: the paper tracker lost ticks to transient transport failures
(``SSL: UNEXPECTED_EOF_WHILE_READING``, DNS resolution hiccups) against a single
hard-coded host. ``get_json``/``post_json`` add what a once-a-day cron job needs:

* exponential backoff with jitter between attempts (``backoff * 2**i`` seconds,
  plus ``U(0, backoff)``, capped at ``max_delay``);
* round-robin over a list of fallback hosts — attempt ``k`` goes to
  ``urls[k % len(urls)]``, and every host is tried at least once even when
  ``tries`` is smaller than the host list;
* explicit failure classes only (``requests.RequestException`` and ``ValueError``
  from JSON decoding); anything else propagates immediately;
* a final ``NetworkError`` carrying one line per failed attempt.

Every failed attempt is logged as a single WARNING line (host + exception
class) through the standard ``logging`` module; the caller decides where the
log goes (the tracker sends it to stdout).

``sleep`` and ``rng`` are injectable so the schedule is testable without
waiting or patching global randomness.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

DEFAULT_TRIES = 4
DEFAULT_BACKOFF = 1.5
DEFAULT_TIMEOUT = 20.0
MAX_DELAY = 30.0

# Module-level so tests can neutralise waiting without touching ``time``.
_sleep: Callable[[float], None] = time.sleep


class NetworkError(Exception):
    """Every attempt against every host failed.

    ``attempts`` holds one human-readable line per failed attempt, in order:
    ``"GET api.binance.com: SSLError: ..."``. The last underlying exception is
    chained as ``__cause__``.
    """

    def __init__(self, message: str, attempts: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.attempts: list[str] = list(attempts)


def backoff_delay(attempt: int, backoff: float, rng: random.Random | Any,
                  max_delay: float = MAX_DELAY) -> float:
    """Seconds to wait before retry number ``attempt`` (0-based).

    ``backoff * 2**attempt`` capped at ``max_delay``, plus uniform jitter in
    ``[0, backoff)`` so that two schedulers firing together do not retry in
    lock-step. ``rng`` only needs a ``uniform(a, b)`` method.
    """
    base = min(backoff * (2.0 ** attempt), max_delay)
    return base + float(rng.uniform(0.0, backoff))


def _normalise_urls(urls: str | Sequence[str]) -> list[str]:
    out = [urls] if isinstance(urls, str) else [u for u in urls if u]
    if not out:
        raise ValueError("at least one URL is required")
    return out


def _host(url: str) -> str:
    return urlsplit(url).netloc or url


def _request_json(method: str, urls: str | Sequence[str], *,
                  params: Mapping[str, Any] | None,
                  body: Any,
                  tries: int,
                  backoff: float,
                  timeout: float,
                  session: requests.Session | None,
                  rng: random.Random | Any | None,
                  sleep: Callable[[float], None] | None) -> Any:
    url_list = _normalise_urls(urls)
    if tries < 1:
        raise ValueError("tries must be >= 1")
    n_attempts = max(tries, len(url_list))
    rng = rng if rng is not None else random.Random()
    do_sleep = sleep if sleep is not None else _sleep

    attempts: list[str] = []
    last_exc: BaseException | None = None
    for i in range(n_attempts):
        url = url_list[i % len(url_list)]
        try:
            if method == "GET":
                get: Callable[..., requests.Response] = (
                    session.get if session is not None else requests.get)
                resp = get(url, params=params, timeout=timeout)
            else:
                post: Callable[..., requests.Response] = (
                    session.post if session is not None else requests.post)
                resp = post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            line = f"{method} {_host(url)}: {type(exc).__name__}: {str(exc)[:200]}"
            attempts.append(line)
            logger.warning("attempt %d/%d failed: %s", i + 1, n_attempts, line)
            if i + 1 < n_attempts:
                do_sleep(backoff_delay(i, backoff, rng))
    raise NetworkError(
        f"{method} {_host(url_list[0])} failed after {n_attempts} attempts "
        f"across {len(url_list)} host(s)", attempts) from last_exc


def get_json(urls: str | Sequence[str], *,
             params: Mapping[str, Any] | None = None,
             tries: int = DEFAULT_TRIES,
             backoff: float = DEFAULT_BACKOFF,
             timeout: float = DEFAULT_TIMEOUT,
             session: requests.Session | None = None,
             rng: random.Random | Any | None = None,
             sleep: Callable[[float], None] | None = None) -> Any:
    """GET the first URL that answers with valid JSON; see module docstring.

    Raises ``NetworkError`` once ``max(tries, len(urls))`` attempts are spent.
    """
    return _request_json("GET", urls, params=params, body=None, tries=tries,
                         backoff=backoff, timeout=timeout, session=session,
                         rng=rng, sleep=sleep)


def post_json(url: str | Sequence[str], body: Any, *,
              tries: int = DEFAULT_TRIES,
              backoff: float = DEFAULT_BACKOFF,
              timeout: float = DEFAULT_TIMEOUT,
              session: requests.Session | None = None,
              rng: random.Random | Any | None = None,
              sleep: Callable[[float], None] | None = None) -> Any:
    """POST ``body`` as JSON, same retry/rotation semantics as ``get_json``.

    ``url`` may be a single URL or a list of fallback URLs.
    """
    return _request_json("POST", url, params=None, body=body, tries=tries,
                         backoff=backoff, timeout=timeout, session=session,
                         rng=rng, sleep=sleep)


__all__ = ["NetworkError", "backoff_delay", "get_json", "post_json",
           "DEFAULT_TRIES", "DEFAULT_BACKOFF", "DEFAULT_TIMEOUT", "MAX_DELAY"]
