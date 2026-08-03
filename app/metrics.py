"""
LLMOps observability layer.

Every API call appends one structured record to logs/requests.jsonl.
`summarize_metrics()` turns that log into the metrics dashboard the
assignment asks for (>= 5 measured metrics).

Metrics served live from the request log:
  M1  Latency          p50 / p95 / p99, overall and per endpoint
  M2  Throughput       requests per minute across the session window
  M3  Reliability      success rate and error rate, with error breakdown
  M4  Token usage      prompt/completion/total tokens + estimated cost
  M5  Degradation rate share of LLM calls that fell back to the local model
  M6  Model confidence mean confidence per endpoint (data-quality signal)

Metric served from the offline harness (needs labels, so it cannot be live):
  M7  Quality          macro-F1 / accuracy, fine-tuned vs prompted baseline
                       -> written by scripts/evaluate.py to logs/eval_report.json

No raw resume text is ever logged. Resumes are personal data; the log keeps
only counts, latencies and model metadata.
"""

import json
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from app import config

# NVIDIA NIM's free tier is not billed, but a cost model is still required for
# LLMOps: it is what makes a token count actionable. Rate below is the public
# reference price for a 20B-class hosted model, per 1M tokens.
COST_PER_1M_TOKENS_USD = 0.10

EVAL_REPORT = config.LOG_DIR / "eval_report.json"


def log_event(
    endpoint: str,
    latency_sec: float,
    success: bool,
    extra: Optional[dict] = None,
) -> None:
    record = {
        "ts": time.time(),
        "endpoint": endpoint,
        "latency_sec": round(latency_sec, 4),
        "success": success,
        "app_version": config.APP_VERSION,
        **(extra or {}),
    }
    with open(config.REQUEST_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile. Correct at n=1 and at pct=100, unlike an
    `int(n * pct) - 1` index, which underflows to -1 on small samples."""
    if not sorted_values:
        return 0.0
    rank = max(1, min(len(sorted_values), int(-(-len(sorted_values) * pct // 100))))
    return sorted_values[rank - 1]


def _latency_stats(values: List[float]) -> Dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "p99": round(_percentile(ordered, 99), 3),
        "mean": round(sum(ordered) / len(ordered), 3) if ordered else 0.0,
        "max": round(ordered[-1], 3) if ordered else 0.0,
    }


def _read_log() -> List[dict]:
    if not config.REQUEST_LOG.exists():
        return []
    records = []
    with open(config.REQUEST_LOG, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line must not break the dashboard
    return records


def summarize_metrics() -> Dict:
    records = _read_log()
    if not records:
        return {"message": "No requests logged yet. Run a demo session first."}

    timestamps = [r["ts"] for r in records]
    window_sec = max(max(timestamps) - min(timestamps), 1e-6)

    by_endpoint = defaultdict(list)
    for record in records:
        by_endpoint[record["endpoint"]].append(record)

    failures = [r for r in records if not r["success"]]
    llm_calls = [r for r in records if r.get("degraded") is not None]
    degraded = [r for r in llm_calls if r.get("degraded")]

    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in records)
    completion_tokens = sum(r.get("completion_tokens") or 0 for r in records)
    total_tokens = sum(r.get("total_tokens") or 0 for r in records)

    confidences = defaultdict(list)
    for record in records:
        if isinstance(record.get("confidence"), (int, float)):
            confidences[record["endpoint"]].append(record["confidence"])

    return {
        "session": {
            "total_requests": len(records),
            "window_sec": round(window_sec, 1),
            "app_version": config.APP_VERSION,
            "llm_backend": config.NVIDIA_MODEL if config.llm_available() else config.FALLBACK_GEN_MODEL,
        },
        "M1_latency_sec": {
            "overall": _latency_stats([r["latency_sec"] for r in records]),
            "by_endpoint": {
                endpoint: _latency_stats([r["latency_sec"] for r in rows])
                for endpoint, rows in sorted(by_endpoint.items())
            },
        },
        "M2_throughput": {
            "requests_per_min": round(len(records) / window_sec * 60, 2),
            "requests_by_endpoint": {ep: len(rows) for ep, rows in sorted(by_endpoint.items())},
        },
        "M3_reliability": {
            "success_rate": round(1 - len(failures) / len(records), 4),
            "error_rate": round(len(failures) / len(records), 4),
            "errors_by_endpoint": dict(Counter(r["endpoint"] for r in failures)),
            "error_types": dict(Counter((r.get("error") or "unknown")[:60] for r in failures)),
        },
        "M4_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_llm_call": round(total_tokens / len(llm_calls), 1) if llm_calls else None,
            "estimated_cost_usd": round(total_tokens / 1_000_000 * COST_PER_1M_TOKENS_USD, 6),
            "cost_model_usd_per_1m_tokens": COST_PER_1M_TOKENS_USD,
        },
        "M5_degradation": {
            "llm_calls": len(llm_calls),
            "fell_back_to_local": len(degraded),
            "degradation_rate": round(len(degraded) / len(llm_calls), 4) if llm_calls else None,
            "note": "Fallback keeps the service available when the LLM API is unreachable.",
        },
        "M6_model_confidence": {
            endpoint: round(sum(values) / len(values), 4)
            for endpoint, values in sorted(confidences.items())
        },
        "M7_quality_offline": _load_eval_report(),
    }


def _load_eval_report() -> Dict:
    if not EVAL_REPORT.exists():
        return {"message": "Run scripts/evaluate.py to populate fine-tuned vs prompted quality metrics."}
    return json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
