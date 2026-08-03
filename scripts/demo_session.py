"""
Drive the canonical demo session, then print the metrics it produced.

    uvicorn app.main:app --port 8000     # in another terminal
    python scripts/demo_session.py       # then this

Why this exists: the LLMOps figures in the report (M1-M6) are computed from
logs/requests.jsonl, which is a live append-only log. Any later run -- a smoke
test, a screenshot capture -- appends to it, so numbers quoted from an ad-hoc
session cannot be reproduced or checked. This script defines one fixed sequence
of requests, so the reported metrics are regenerable rather than a snapshot of
a session nobody can reconstruct.

It rotates the existing log aside first, so the metrics describe exactly this
session and nothing else.

Two of the requests fail deliberately: an unsupported file type (415) and an
oversized upload (413). A reliability metric that only ever sees successes
measures nothing -- M3 needs real errors to be worth reporting.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RESUME = ROOT / "data" / "sample_resume.pdf"
JD_PATH = ROOT / "data" / "sample_jd.txt"
LOG = ROOT / "logs" / "requests.jsonl"

QUESTIONS = [
    "What programming languages does the candidate know?",
    "How many years of experience does the candidate have?",
    # No cloud certification is claimed in a form the model can quote, so this
    # one is expected to abstain. An abstention in the transcript is the point:
    # it shows the QA endpoint declines rather than inventing a span.
    "What is the candidate's expected salary?",
]


def rotate_log() -> None:
    if LOG.exists():
        backup = LOG.with_suffix(".jsonl.prev")
        shutil.move(str(LOG), str(backup))
        print(f"rotated previous log -> {backup.name}")
    LOG.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical demo session")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--keep-log", action="store_true",
                        help="append to the existing log instead of rotating it")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    if not RESUME.exists():
        raise SystemExit(f"missing {RESUME} -- run scripts/make_sample_resume.py")

    try:
        requests.get(f"{base}/health", timeout=10).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"server not reachable at {base}: {exc}")

    if not args.keep_log:
        rotate_log()

    jd = JD_PATH.read_text(encoding="utf-8")
    pdf = RESUME.read_bytes()
    started = time.time()

    def call(label: str, method: str, path: str, **kwargs):
        t0 = time.time()
        response = requests.request(method, f"{base}{path}", timeout=600, **kwargs)
        elapsed = time.time() - t0
        print(f"  {label:<34} {response.status_code}  {elapsed:6.2f}s")
        return response

    print(f"canonical session against {base}\n")

    call("CV1  /classify-document", "POST", "/classify-document",
         files={"file": ("sample_resume.pdf", pdf, "application/pdf")})
    call("CV2  /extract-text", "POST", "/extract-text",
         files={"file": ("sample_resume.pdf", pdf, "application/pdf")})
    ingest = call("CV1+2 /ingest-resume", "POST", "/ingest-resume",
                  files={"file": ("sample_resume.pdf", pdf, "application/pdf")})
    text = ingest.json()["extraction"]["text"]

    call("NLP3 /entities", "POST", "/entities", json={"text": text})
    call("NLP4 /classify-fit", "POST", "/classify-fit",
         json={"resume_text": text, "job_description": jd})
    call("NLP4 /compare-fit-models", "POST", "/compare-fit-models",
         json={"resume_text": text, "job_description": jd})

    for question in QUESTIONS:
        call("NLP5 /ask", "POST", "/ask",
             json={"resume_text": text, "question": question})

    call("NLP6 /candidate-brief", "POST", "/candidate-brief",
         json={"resume_text": text, "job_description": jd})
    call("ALL  /screen-candidate", "POST", "/screen-candidate",
         json={"resume_text": text, "job_description": jd})

    # Deliberate failures -- see the module docstring.
    call("ERR  415 unsupported type", "POST", "/ingest-resume",
         files={"file": ("deploy.sh", b"#!/bin/sh\necho hi\n", "text/x-shellscript")})
    call("ERR  413 oversized upload", "POST", "/ingest-resume",
         files={"file": ("huge.pdf", b"%PDF-1.4\n" + b"0" * (12 * 1024 * 1024),
                         "application/pdf")})

    print(f"\nsession wall clock: {time.time() - started:.1f}s")

    summary = requests.get(f"{base}/metrics", timeout=60).json()
    out = ROOT / "logs" / "metrics_snapshot.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}\n")

    session = summary["session"]
    print(f"M0  requests           {session['total_requests']} over {session['window_sec']}s")
    print(f"M1  latency p50/p95    {summary['M1_latency_sec']['overall']['p50']}s / "
          f"{summary['M1_latency_sec']['overall']['p95']}s")
    print(f"M2  throughput         {summary['M2_throughput']['requests_per_min']} req/min")
    print(f"M3  success rate       {summary['M3_reliability']['success_rate']:.2%}")
    tokens = summary["M4_token_usage"]
    print(f"M4  tokens             {tokens['prompt_tokens']} prompt + "
          f"{tokens['completion_tokens']} completion, {tokens['total_tokens']} total, "
          f"${tokens['estimated_cost_usd']}")
    print(f"M5  degradation        {summary['M5_degradation']['degradation_rate']}")
    for endpoint, value in summary["M6_model_confidence"].items():
        print(f"M6  confidence         {endpoint} {value}")

    # The report quotes prompt, completion and total side by side, so they have
    # to agree. They only do if every LLM-calling endpoint logs all three: an
    # endpoint logging just the total inflates it above the parts it is summed
    # against. Assert rather than print -- a silently broken identity in the
    # metrics section discredits every other number in the report.
    gap = tokens["total_tokens"] - tokens["prompt_tokens"] - tokens["completion_tokens"]
    assert gap == 0, (
        f"M4 inconsistent: prompt {tokens['prompt_tokens']} + completion "
        f"{tokens['completion_tokens']} != total {tokens['total_tokens']} "
        f"(off by {gap}). An LLM endpoint is logging total_tokens without its parts."
    )
    print(f"\n  M4 identity holds: {tokens['prompt_tokens']} + "
          f"{tokens['completion_tokens']} = {tokens['total_tokens']}")


if __name__ == "__main__":
    main()
