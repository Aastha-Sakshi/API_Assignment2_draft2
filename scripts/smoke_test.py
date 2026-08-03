"""
End-to-end smoke test — exercises every sub-task against the real models.

    python scripts/smoke_test.py

First run downloads ~2 GB of model weights. Sub-tasks whose prerequisites are
missing (fine-tuned model, API key) are reported as SKIP rather than failing
the run.
"""

import io
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESUME = (ROOT / "data" / "sample_resume.txt").read_text(encoding="utf-8")
JD = (ROOT / "data" / "sample_jd.txt").read_text(encoding="utf-8")

results = []


def check(name, fn):
    try:
        value = fn()
        print(f"  PASS  {name}: {value}")
        results.append(("PASS", name))
    except Exception as exc:
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        results.append(("FAIL", name))


def skip(name, reason):
    print(f"  SKIP  {name}: {reason}")
    results.append(("SKIP", name))


def render_resume_png() -> bytes:
    """A synthetic scan, so the CV path can be tested without a sample image."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    y = 60
    for line in RESUME.splitlines():
        for wrapped in textwrap.wrap(line, 90) or [""]:
            draw.text((70, y), wrapped, fill="black")
            y += 26
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


print("\n--- CV 1: document-type classification (DiT) ---")
check(
    "classify_document_type",
    lambda: (lambda r: f"{r['predicted_type']} @ {r['confidence']:.2f}, {r['latency_sec']}s")(
        pipeline.classify_document_type(render_resume_png())
    ),
)

print("\n--- CV 2: text extraction ---")
check(
    "extract_text(txt)",
    lambda: (lambda r: f"{r['extraction_method']}, {r['char_count']} chars")(
        pipeline.extract_text(RESUME.encode(), ".txt")
    ),
)

check(
    "extract_text(png -> docTR OCR)",
    lambda: (lambda r: f"{r['extraction_method']}, {r['char_count']} chars, {r['latency_sec']}s")(
        pipeline.extract_text(render_resume_png(), ".png")
    ),
)

print("\n--- NLP 3: NER ---")
check(
    "extract_entities",
    lambda: (lambda r: f"{r['count']} entities, groups={list(r['grouped'])}, {r['latency_sec']}s")(
        pipeline.extract_entities(RESUME)
    ),
)

print("\n--- NLP 4: fit classification ---")
if pipeline.finetuned_available():
    check(
        "classify_fit_finetuned",
        lambda: (lambda r: f"{r['label']} @ {r['confidence']:.2f}, {r['latency_sec']}s")(
            pipeline.classify_fit_finetuned(RESUME, JD)
        ),
    )
else:
    skip("classify_fit_finetuned", f"no model at {config.FINETUNED_DIR} — run finetune_classifier.py")

if config.llm_available():
    check(
        "classify_fit_prompted",
        lambda: (lambda r: f"{r['label']} via {r['model']['model']}, {r['latency_sec']}s")(
            pipeline.classify_fit_prompted(RESUME, JD)
        ),
    )
else:
    skip("classify_fit_prompted", "NVIDIA_API_KEY not set")

print("\n--- NLP 5: extractive QA ---")
check(
    "answer_question",
    lambda: (lambda r: f"'{r['answer']}' @ {r['confidence']:.2f}, {r['latency_sec']}s")(
        pipeline.answer_question(RESUME, "Which cloud platform has the candidate used?")
    ),
)

print("\n--- NLP 6: candidate brief ---")
check(
    "generate_candidate_brief",
    lambda: (lambda r: f"{len(r['brief'])} chars via {r['backend']}, degraded={r['degraded']}, {r['latency_sec']}s")(
        pipeline.generate_candidate_brief(RESUME, JD)
    ),
)

print("\n--- orchestration ---")
check(
    "screen_candidate",
    lambda: (lambda r: f"verdict={r['fit']['label']}, {r['pipeline_latency_sec']}s total")(
        pipeline.screen_candidate(RESUME, JD, include_brief=False)
    ),
)

passed = sum(1 for status, _ in results if status == "PASS")
failed = sum(1 for status, _ in results if status == "FAIL")
skipped = sum(1 for status, _ in results if status == "SKIP")
print(f"\n{passed} passed · {failed} failed · {skipped} skipped")
sys.exit(1 if failed else 0)
