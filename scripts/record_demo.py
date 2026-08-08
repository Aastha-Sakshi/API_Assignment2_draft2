"""
Record the ~5 minute screen demo, driven end to end.

    uvicorn app.main:app --port 8000          # terminal 1
    streamlit run streamlit_app.py            # terminal 2
    python scripts/record_demo.py             # then this

Produces docs/demo.mp4: a silent 1920x1080 recording of the whole pipeline,
paced for a voice-over to be laid over it afterwards. Nothing is sped up and no
cuts are made -- the pauses are real, so what the video shows is what the
service actually does, including how much longer the LLM takes than the
fine-tuned classifier.

It records the *page*, via Playwright, not the desktop. That matters for two
reasons. Desktop capture puts whatever sits behind the browser into the video --
editor, terminal, notifications -- which is a privacy problem for a file meant
to be shared. And a headed browser window has to be maximized to cover the
screen, which races page load and intermittently records a half-sized window.
Recording the page sidesteps both: the frame is exactly the viewport, always,
and nothing outside the app can appear in it.

ffmpeg is used only to convert Playwright's .webm into an .mp4 that Word,
PowerPoint and QuickTime will play.

    --scale 1.3     stretch every pause by 30% for a slower voice-over
    --out other.mp4 write somewhere else
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"
UI = "http://127.0.0.1:8501"

WIDTH, HEIGHT = 1920, 1080

# The fabricated resume, never a real CV: this video is meant to be shared, and
# the UI renders the contact block in full.
RESUME = ROOT / "data" / "sample_resume.pdf"
JD = ROOT / "data" / "sample_jd.txt"

SCALE = 1.0  # set from --scale


def beat(page, seconds: float, note: str = "") -> None:
    """Hold the current view long enough to narrate over it."""
    if note:
        print(f"    {note}  ({seconds * SCALE:.0f}s)")
    page.wait_for_timeout(int(seconds * SCALE * 1000))


def tab(page, index: int, settle: float = 2.0) -> None:
    page.locator('button[role="tab"]').nth(index).click()
    beat(page, settle)


def run_button(page, label: str, budget: int = 240) -> float:
    """Click, wait for Streamlit to go idle, return how long it really took."""
    button = page.get_by_role("button", name=label, exact=False).first
    button.scroll_into_view_if_needed()
    page.wait_for_timeout(600)
    started = time.time()
    button.click()

    # Wait for the RUNNING indicator to APPEAR before waiting for it to go.
    # Checking only for its absence returns instantly when Streamlit has not
    # rendered it yet, so a slow call gets "finished" while still in flight and
    # the recording captures a spinner instead of a result.
    status = page.locator('[data-testid="stStatusWidget"]')
    appeared_by = time.time() + 8
    while time.time() < appeared_by and status.count() == 0:
        page.wait_for_timeout(150)

    deadline = time.time() + budget
    while time.time() < deadline:
        if status.count() == 0:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(1200)  # let the result paint
    elapsed = time.time() - started
    print(f"    '{label}' took {elapsed:.1f}s")
    return elapsed


def scroll(page, amount: int, steps: int = 6, pause: float = 0.45) -> None:
    """Smooth scroll -- one big jump is unreadable on video."""
    for _ in range(steps):
        page.mouse.wheel(0, amount // steps)
        page.wait_for_timeout(int(pause * 1000))


def convert(webm: Path, out: Path) -> None:
    """webm -> mp4. yuv420p and even dimensions or Office refuses to play it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(webm),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-movflags", "+faststart", str(out)],
        check=True,
    )


def demo(page, jd_text: str) -> None:
    """The scripted walkthrough. Ordered to match the report's sections."""
    # 1 -- the API surface (requirement 6) ---------------------------------
    print("  [1/12] Swagger: the API surface")
    page.goto(f"{API}/docs", wait_until="networkidle", timeout=120_000)
    beat(page, 9, "title + endpoint list")
    scroll(page, 900)
    beat(page, 7, "CV and NLP endpoint groups")
    scroll(page, 900)
    beat(page, 7, "orchestration + LLMOps groups")
    scroll(page, -1800, steps=4, pause=0.3)
    beat(page, 3)

    # 2 -- model registry ---------------------------------------------------
    print("  [2/12] Model registry")
    page.goto(f"{API}/model-registry", wait_until="networkidle")
    beat(page, 14, "every model, by sub-task and category")

    # 3 -- health -----------------------------------------------------------
    print("  [3/12] Health")
    page.goto(f"{API}/health", wait_until="networkidle")
    beat(page, 10, "backend reachable, fine-tuned model loaded")

    # 4 -- the UI -----------------------------------------------------------
    print("  [4/12] Streamlit home")
    page.goto(UI, wait_until="networkidle", timeout=120_000)
    page.wait_for_timeout(6000)
    beat(page, 10, "one tab per sub-task; the client holds no model code")

    # 5 -- ingest -----------------------------------------------------------
    print("  [5/12] Upload + ingest")
    tab(page, 0)
    page.locator('input[type="file"]').set_input_files(str(RESUME))
    beat(page, 4, "file selected")
    run_button(page, "Ingest document")
    beat(page, 12, "DiT confirms a resume, then the text layer is read")
    scroll(page, 600, steps=4)
    beat(page, 8, "extracted text")

    jd_box = page.get_by_placeholder("Paste the JD here", exact=False)
    if jd_box.count():
        jd_box.first.fill(jd_text)
        jd_box.first.blur()
        beat(page, 6, "job description pasted")
    else:
        print("    WARNING: JD box not found")

    # 6 -- entities ---------------------------------------------------------
    print("  [6/12] NER")
    tab(page, 1)
    run_button(page, "Extract entities")
    beat(page, 14, "skills/titles/dates scored; identifiers held back")

    # 7 -- fine-tuned classifier --------------------------------------------
    print("  [7/12] Fit — fine-tuned")
    tab(page, 2)
    run_button(page, "Run fine-tuned model")
    beat(page, 14, "label + full score distribution, ~1s on CPU")

    # 8 -- the comparison, the report's centrepiece -------------------------
    print("  [8/12] Fit — both models side by side")
    run_button(page, "Compare both side by side")
    beat(page, 18, "they disagree; note the latency gap")
    scroll(page, 500, steps=4)
    beat(page, 8)

    # 9 -- extractive QA -----------------------------------------------------
    print("  [9/12] Question answering")
    tab(page, 3)
    question = page.get_by_label("Question")
    if not question.count():
        question = page.locator('input[type="text"]').last
    question.first.fill("What programming languages does the candidate know?")
    beat(page, 3)
    run_button(page, "Ask")
    beat(page, 12, "a literal span from the resume, not a paraphrase")

    question.first.fill("What is the candidate's expected salary?")
    beat(page, 3, "a question the resume cannot answer")
    run_button(page, "Ask")
    beat(page, 12, "it declines instead of inventing one")

    # 10 -- generation --------------------------------------------------------
    print("  [10/12] Candidate brief")
    tab(page, 4)
    run_button(page, "Generate brief")
    beat(page, 16, "written from PII-redacted text")
    scroll(page, 700, steps=5)
    beat(page, 12, "strengths, gaps, interview questions")

    # 11 -- the whole chain in one call ---------------------------------------
    print("  [11/12] Full screening")
    tab(page, 5)
    run_button(page, "Screen candidate")
    beat(page, 16, "every sub-task in a single request")
    scroll(page, 800, steps=5)
    beat(page, 12)

    # 12 -- LLMOps ------------------------------------------------------------
    print("  [12/12] LLMOps")
    tab(page, 6)
    run_button(page, "Refresh /metrics", budget=90)
    beat(page, 14, "M1-M6 live from the request log")
    scroll(page, 700, steps=5)
    beat(page, 14, "latency, reliability, tokens, cost, confidence")
    scroll(page, 700, steps=5)
    beat(page, 10)


def main() -> None:
    global SCALE
    parser = argparse.ArgumentParser(description="Record the demo video")
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo.mp4"))
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every pause (1.3 = 30%% slower)")
    parser.add_argument("--headed", action="store_true",
                        help="show the browser while it records (the recording "
                             "is identical either way -- it captures the page)")
    args = parser.parse_args()
    SCALE = args.scale
    out = Path(args.out)

    if not RESUME.exists():
        raise SystemExit(f"missing {RESUME} -- run scripts/make_sample_resume.py")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH -- needed to write the .mp4")

    jd_text = JD.read_text(encoding="utf-8")
    staging = Path(tempfile.mkdtemp(prefix="demo_video_"))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=not args.headed)
        # Headless by default: the frame is the viewport, so it is exactly
        # 1920x1080 regardless of the physical screen, and no window ever has
        # to be maximized.
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=str(staging),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        page = context.new_page()
        started = time.time()
        try:
            demo(page, jd_text)
        finally:
            elapsed = time.time() - started
            # close() flushes and finalizes the .webm; without it the file is
            # truncated and the last scene is missing.
            context.close()
            browser.close()

    videos = list(staging.glob("*.webm"))
    if not videos:
        raise SystemExit(f"playwright wrote no video into {staging}")
    print(f"\n  converting {videos[0].name} -> mp4")
    convert(videos[0], out)
    shutil.rmtree(staging, ignore_errors=True)

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  recorded {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    if elapsed < 240:
        print("  under 4 min -- re-run with --scale 1.3 for a slower read")
    elif elapsed > 400:
        print("  over 6.5 min -- re-run with --scale 0.85")


if __name__ == "__main__":
    main()
