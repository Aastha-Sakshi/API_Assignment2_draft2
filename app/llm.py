"""
LLM access layer.

Primary: NVIDIA NIM (build.nvidia.com) — OpenAI-compatible, free tier,
returns real token counts so the LLMOps dashboard has cost/usage data.
Fallback: local google/flan-t5-base, so the demo never dies on a missing key
or a dead network during the viva.

Prompts are versioned. In LLMOps a prompt is a deployed artifact: if the
output changes, you must be able to say which prompt produced it.
"""

import time
import logging
from functools import lru_cache

from app import config

logger = logging.getLogger("llm")

PROMPT_VERSIONS = {
    "candidate_brief": "v3",
    "fit_classify": "v2",
}


class LLMResult(dict):
    """dict subclass so it serialises straight to JSON via FastAPI."""


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(
        base_url=config.NVIDIA_BASE_URL,
        api_key=config.NVIDIA_API_KEY,
        timeout=config.LLM_TIMEOUT_SEC,
    )


@lru_cache(maxsize=1)
def _fallback_pipe():
    from transformers import pipeline

    return pipeline("text2text-generation", model=config.FALLBACK_GEN_MODEL)


# gpt-oss-20b is a reasoning model: it spends completion tokens on hidden
# reasoning before emitting any content. A tight max_tokens returns an EMPTY
# string with finish_reason="length" — a silent failure that looks like a model
# refusing to answer.
#
# Measured on real resume+JD pairs: 288 tokens left 1 of 3 empty, 768 answered
# all 3. Raising the ceiling is close to free — a reasoning model stops when it
# is done, so total_tokens barely moved between a 768 and a 1536 budget (1506 vs
# 1512 on the same row). Cheap insurance against a silent blank.
REASONING_TOKEN_HEADROOM = 1024


_reasoning_effort_supported = True


def _create(prompt: str, max_tokens: int, temperature: float):
    """
    One completion call, with reasoning_effort when the provider accepts it.

    The parameter is not universally supported, and llm.chat treats any
    exception as "the API is down" and degrades to flan-t5. Letting an unknown
    keyword trigger that would silently swap the model out for a much weaker one
    over a param the task does not need, so a rejection is caught here, recorded
    once, and the call is retried plainly.
    """
    global _reasoning_effort_supported

    kwargs = dict(
        model=config.NVIDIA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if config.LLM_REASONING_EFFORT and _reasoning_effort_supported:
        try:
            return _client().chat.completions.create(
                **kwargs, extra_body={"reasoning_effort": config.LLM_REASONING_EFFORT}
            )
        except Exception as exc:
            if not _is_bad_request(exc):
                raise  # a real outage: let chat() degrade as designed
            _reasoning_effort_supported = False
            logger.warning(
                "%s rejected reasoning_effort=%s (%s) — disabling it for this process",
                config.NVIDIA_MODEL, config.LLM_REASONING_EFFORT, str(exc)[:120],
            )

    return _client().chat.completions.create(**kwargs)


def _is_bad_request(exc: Exception) -> bool:
    """A 4xx that is not auth/rate-limit means the request itself was rejected."""
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (401, 403, 429)


def chat(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> LLMResult:
    """
    Single-turn completion. Always returns a result — degrades to the local
    model rather than raising, and says which path it took.
    """
    start = time.perf_counter()

    if config.llm_available():
        try:
            headroom = REASONING_TOKEN_HEADROOM
            for attempt in (1, 2):
                resp = _create(prompt, max_tokens + headroom, temperature)
                choice = resp.choices[0]
                usage = resp.usage
                text = (choice.message.content or "").strip()
                if text or choice.finish_reason != "length":
                    break
                # Ran out of budget mid-reasoning. Retry once at double, because
                # scoring this as a wrong answer would blame the model for an
                # infrastructure limit and understate the prompted arm.
                logger.warning(
                    "empty content from %s (finish_reason=%s, completion_tokens=%s) "
                    "— attempt %d of 2 at max_tokens=%d",
                    config.NVIDIA_MODEL, choice.finish_reason,
                    getattr(usage, "completion_tokens", None), attempt, max_tokens + headroom,
                )
                headroom *= 2

            if not text:
                # Both attempts produced nothing. Returning an empty string with
                # degraded=False would ship a blank brief as a success and hide
                # the event from M5, which counts fallbacks. Raising routes it
                # through the same path as a network failure: the local model
                # answers, and the degradation is recorded.
                raise RuntimeError(
                    f"{config.NVIDIA_MODEL} returned no content in 2 attempts "
                    f"(finish_reason={choice.finish_reason}, budget={max_tokens + headroom // 2})"
                )

            return LLMResult(
                text=text,
                model=config.NVIDIA_MODEL,
                backend="nvidia-nim",
                degraded=False,
                # A response cut off mid-sentence is truncated whether or not any
                # text survived; reporting only the empty case let a half-written
                # brief through as complete.
                truncated=choice.finish_reason == "length",
                finish_reason=choice.finish_reason,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                latency_sec=round(time.perf_counter() - start, 3),
            )
        except Exception as exc:  # network, rate limit, bad key
            logger.warning("NIM call failed (%s) — falling back to %s", exc, config.FALLBACK_GEN_MODEL)
            degraded_reason = str(exc)[:200]
    else:
        degraded_reason = "NVIDIA_API_KEY not set"

    # ponytail: flan-t5-base has a 512-token window, so the fallback truncates.
    # It keeps the demo alive; it is not expected to match gpt-oss-20b quality.
    out = _fallback_pipe()(prompt[:1800], max_new_tokens=min(max_tokens, 256))
    return LLMResult(
        text=out[0]["generated_text"].strip(),
        model=config.FALLBACK_GEN_MODEL,
        backend="local-transformers",
        degraded=True,
        degraded_reason=degraded_reason,
        truncated=False,
        finish_reason="stop",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        latency_sec=round(time.perf_counter() - start, 3),
    )
