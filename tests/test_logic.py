"""
Self-checks for the non-obvious logic. No test framework, no fixtures:

    python tests/test_logic.py

Covers only the parts that fail silently and wrongly if broken — label
parsing, chunking, percentiles, text cleanup. Model inference is not tested
here; that is what scripts/evaluate.py is for.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import metrics, pipeline  # noqa: E402


def test_parse_fit_label():
    parse = pipeline._parse_fit_label

    assert parse("Good Fit") == "Good Fit"
    assert parse("no fit") == "No Fit"
    assert parse("The candidate is a Potential Fit for this role.") == "Potential Fit"
    assert parse("banana") is None
    assert parse("") is None

    # The one that matters: "Potential Fit" contains no substring of "No Fit",
    # but a naive shortest-first scan over "Good Fit" style outputs can still
    # mis-order. Longest-first must win.
    assert parse("Label: Potential Fit") == "Potential Fit"
    assert parse("Potential Fit\nNo Fit") == "Potential Fit"


def test_chunking_covers_all_text():
    lines = [f"line {i} with some content" for i in range(200)]
    text = "\n".join(lines)

    chunks = pipeline._chunk(text, max_chars=500)

    assert len(chunks) > 1, "long text must be split"
    # Nothing may be dropped — the old code truncated at 2000 chars silently.
    rejoined = "\n".join(chunks)
    assert rejoined.count("line ") == 200
    for chunk in chunks:
        assert len(chunk) <= 500 + 40  # one overshooting line is tolerated

    assert pipeline._chunk("") == [""]
    assert pipeline._chunk("short") == ["short"]


def test_clean_text_keeps_line_structure():
    raw = "Skills: Python  \t Java\n\n\n\nExperience:\n- built things  "
    cleaned = pipeline.clean_text(raw)

    assert " " not in cleaned
    assert "Skills: Python Java" in cleaned
    assert "\n\n\n" not in cleaned
    assert "\n" in cleaned, "line breaks carry section structure — must survive"


def test_redact_pii():
    """
    The responsible-use claim in the report rests on this function, so it is
    checked in both directions: identifiers must go, and ordinary numbers must
    stay. An earlier version passed a naive "did the count go up" check while
    leaving the email and phone in the prompt — NER returns spans rebuilt from
    wordpieces ("abc99 @ xyz. com"), which never match the document verbatim.
    """
    text = (
        "Jane Doe\njane.doe99@example.com | +91-9876543210 | linkedin.com/in/janedoe\n"
        "Pune, India\nGraduated 2021 with 8.42 CGPA. Scaled service to 500000 users.\n"
        "Reduced p99 latency from 250 ms to 120 ms."
    )
    detected = [
        {"entity": "NAME", "text": "Jane Doe"},
        {"entity": "LOCATION", "text": "Pune, India"},
        {"entity": "NAME", "text": "##ne"},  # subword fragment: never literal
    ]
    out, count = pipeline.redact_pii(text, detected)

    for gone in ("jane.doe99@example.com", "9876543210", "janedoe", "Jane Doe", "Pune, India"):
        assert gone not in out, f"identifier survived redaction: {gone}"
    for placeholder in ("[EMAIL]", "[PHONE]", "[URL]", "[NAME]", "[LOCATION]"):
        assert placeholder in out, f"missing {placeholder}: {out}"
    assert count >= 5

    # Over-redaction is the quieter failure: it degrades the brief without
    # looking like a bug. Years, CGPAs, counts and timings must survive.
    for kept in ("2021", "8.42", "500000", "250 ms", "120 ms", "p99"):
        assert kept in out, f"over-redacted {kept}: {out}"

    assert pipeline.redact_pii("no identifiers here", []) == ("no identifiers here", 0)


def test_qa_answer_selection():
    """
    Regression for the abstention bug: on a long resume the null answer scores
    0.735 while the correct span scores 0.0135, so ranking by raw score alone
    made the endpoint abstain on every question.
    """
    spans = [
        {"answer": "", "score": 0.735, "start": 0, "end": 0},
        {"answer": "Python, SQL", "score": 0.0135, "start": 10, "end": 21},
    ]
    real = [r for r in spans if r["answer"].strip()]
    best = max(real, key=lambda r: r["score"], default=None)
    assert best is not None and best["score"] >= pipeline.QA_MIN_SPAN_SCORE, \
        "a genuine span above threshold must beat a high-scoring null"

    # Genuinely unanswerable: every real span is below threshold -> abstain.
    weak = [{"answer": "", "score": 0.82, "start": 0, "end": 0},
            {"answer": "+91-98765", "score": 0.0006, "start": 5, "end": 14}]
    real = [r for r in weak if r["answer"].strip()]
    best = max(real, key=lambda r: r["score"], default=None)
    assert best["score"] < pipeline.QA_MIN_SPAN_SCORE, "weak span must abstain"

    # All-null: max(..., default=None) must not raise.
    assert max([r for r in [{"answer": "", "score": 0.9}] if r["answer"].strip()],
               key=lambda r: r["score"], default=None) is None


def test_percentiles():
    pct = metrics._percentile

    assert pct([], 50) == 0.0
    assert pct([1.0], 95) == 1.0, "n=1 must not index -1"
    assert pct([1.0, 2.0], 50) == 1.0
    hundred = [float(i) for i in range(1, 101)]
    assert pct(hundred, 50) == 50.0
    assert pct(hundred, 95) == 95.0
    assert pct(hundred, 99) == 99.0
    assert pct(hundred, 100) == 100.0


def test_latency_stats():
    stats = metrics._latency_stats([0.5, 0.1, 0.9, 0.3])

    assert stats["count"] == 4
    assert stats["max"] == 0.9
    assert stats["mean"] == 0.45
    assert stats["p50"] <= stats["p95"] <= stats["p99"] <= stats["max"]


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} checks passed")
