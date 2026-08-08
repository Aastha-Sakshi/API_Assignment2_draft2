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
import html
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"
UI = "http://127.0.0.1:8501"

WIDTH, HEIGHT = 1920, 1080

# Seconds of blank screen recorded before the walkthrough starts, and cut back
# off in convert(). Absorbs the compositor's start-up frames. The clock starts
# after it, so the cue sheet needs no adjusting.
WARMUP = 4.0

# The fabricated resume, never a real CV: this video is meant to be shared, and
# the UI renders the contact block in full.
RESUME = ROOT / "data" / "sample_resume.pdf"
JD = ROOT / "data" / "sample_jd.txt"

SCALE = 1.0  # set from --scale


CARD_JS = """
([num, title, sub]) => {
  // Idempotent. card() draws, runs the transition, then draws again -- and when
  // that transition is a tab click rather than a navigation, nothing tore the
  // first card down. Two cards with one id meant getElementById() removed one
  // and left the other welded over the app for the rest of the recording.
  document.querySelectorAll('#__demo_card__').forEach(n => n.remove());
  const el = document.createElement('div');
  el.id = '__demo_card__';
  el.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:#0f1420;'
    + 'display:flex;align-items:center;justify-content:center;'
    // pointer-events:none so the card is purely visual. The tab click for the
    // next section happens underneath it, and a card that swallowed clicks
    // would block the very transition it is covering.
    + 'pointer-events:none;'
    + "font-family:'Segoe UI',system-ui,sans-serif;color:#e8ecf4";
  const box = document.createElement('div');
  box.style.cssText = 'max-width:1250px;padding:0 90px';
  const n = document.createElement('div');
  n.style.cssText = "font:600 30px/1 'Segoe UI';letter-spacing:.32em;color:#5b8def";
  n.textContent = num;
  const rule = document.createElement('div');
  rule.style.cssText = 'height:3px;width:82px;background:#5b8def;margin:26px 0 34px';
  const h = document.createElement('div');
  h.style.cssText = "font:600 78px/1.12 'Segoe UI';letter-spacing:-.015em";
  h.textContent = title;
  const s = document.createElement('div');
  s.style.cssText = "font:300 33px/1.5 'Segoe UI';color:#93a2bd;margin-top:26px";
  s.textContent = sub;
  box.append(n, rule, h, s);
  el.append(box);
  // Not document.body: after a "commit" navigation the document exists but the
  // parser may not have reached <body> yet, and appending to null throws --
  // which would kill a recording minutes in. A position:fixed child of <html>
  // renders identically and survives <body> arriving afterwards.
  (document.body || document.documentElement).append(el);
}
"""

CARD_REMOVE_JS = ("() => document.querySelectorAll('#__demo_card__')"
                  ".forEach(n => n.remove())")

# Runs before paint on every new document. Without it a navigation shows the
# browser's white default for as long as the page takes to load -- several
# seconds on Swagger -- which reads as a blank flash between card and content.
# Painting the card's own colour instead makes the gap invisible.
DARK_BG_JS = """
() => {
  const paint = () => {
    document.documentElement.style.background = '#0f1420';
    if (document.body) document.body.style.background = '#0f1420';
  };
  paint();
  document.addEventListener('DOMContentLoaded', paint);
}
"""

TIMELINE = []  # (seconds_from_start, section title) -- written out for the VO script
_started = None


def card(page, num: str, title: str, sub: str, hold: float = 2.6, then=None,
         settle: float = 0.0) -> None:
    """
    A full-frame section card, drawn by the browser itself.

    Drawing the card into the page rather than splicing it in with ffmpeg
    afterwards means it lands exactly where it belongs -- no cut points to
    detect, no drift between the card and the section it introduces.

    It is an overlay on top of the live page, not a replacement for it:
    set_content() would tear down the running Streamlit app, taking the
    uploaded resume and everything else in its session state with it, and every
    tab click after the card would then have nothing to click.

    `then` is the work that moves to the section's content -- a navigation, or a
    tab click. It runs *underneath* the card, which is then re-drawn because a
    navigation destroys the overlay with the rest of the document. Doing it this
    way is the whole point: the card has to introduce its section, and the
    viewer must never see the content before the title that names it.
    """
    if _started is not None:
        TIMELINE.append((time.time() - _started, f"{num} — {title}"))
    print(f"\n  == {num}  {title}")
    args = [num, title, sub]
    page.evaluate(CARD_JS, args)
    page.wait_for_timeout(int(hold * SCALE * 0.55 * 1000))

    if then is not None:
        then()
        # Re-draw immediately: if `then` navigated, the overlay went with the
        # old document. The navigations return at "commit", so this runs before
        # the new page has painted anything -- which is the point. Waiting for
        # "load" there instead left Chromium's blank white document on screen
        # for the whole load, a white flash between the card and its section.
        try:
            page.evaluate(CARD_JS, args)
        except Exception as exc:  # a redirect can swap the document under us
            print(f"    (card re-draw skipped: {type(exc).__name__})")
        # Now let the load finish, behind the card. Bounded: Swagger holds
        # connections open, and a card stuck on screen for a full default
        # timeout would be worse than a section that arrives half-painted.
        try:
            page.wait_for_load_state("load", timeout=20_000)
        except PWTimeout:
            print("    (load did not settle in 20s -- continuing)")

    # Outside the `then` block: a page that keeps drawing after "load" --
    # Swagger fetches its schema, Streamlit builds its layout in JS -- needs the
    # card held longer, and so would a section that is slow for its own reasons.
    page.wait_for_timeout(int(settle * 1000))
    page.wait_for_timeout(int(hold * SCALE * 0.45 * 1000))
    page.evaluate(CARD_REMOVE_JS)
    # A card left behind covers everything after it, and the recording still
    # runs to completion and writes a file -- the failure is only visible by
    # watching all seven minutes. Fail here instead.
    left = page.evaluate("() => document.querySelectorAll('#__demo_card__').length")
    assert left == 0, f"card {num} still on screen after removal ({left} left)"
    page.wait_for_timeout(150)


def click_tab(page, index: int) -> None:
    page.locator('button[role="tab"]').nth(index).click()
    page.wait_for_timeout(900)


def json_render(page, endpoint: str) -> None:
    """
    Show a JSON endpoint at a size that survives video compression.

    The browser's own JSON view renders at ~13px and does not wrap, which is
    unreadable once it has been through h264 at 1080p. Pretty-print it into
    large monospace instead -- same bytes, legible on screen.
    """
    raw = urllib.request.urlopen(API + endpoint, timeout=120).read().decode()
    pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    escaped = html.escape(pretty)
    # Colour keys, strings, numbers and booleans so the structure reads at a
    # glance. Order matters: keys are matched before bare strings.
    escaped = re.sub(r'(&quot;[^&]*?&quot;)(\s*:)',
                     r'<span style="color:#7fb2ff">\1</span>\2', escaped)
    escaped = re.sub(r':\s(&quot;[^&]*?&quot;)',
                     r': <span style="color:#8fd694">\1</span>', escaped)
    escaped = re.sub(r':\s(-?\d+\.?\d*|true|false|null)',
                     r': <span style="color:#f0b45e">\1</span>', escaped)
    page.set_content(
        "<html><body style=\"margin:0;background:#0f1420;\">"
        "<div style=\"background:#161d2e;color:#7fb2ff;"
        "font:600 32px/1 Consolas,monospace;padding:22px 44px;"
        "border-bottom:2px solid #2a3550\">GET " + endpoint + "</div>"
        # 27px, not the 21px this started at: the video gets watched in a
        # window a third of the screen wide, where 21px stops being legible.
        # Fewer lines fit per screen, which the scroll already handles.
        "<pre style=\"color:#dbe3f0;font:27px/1.6 Consolas,monospace;"
        "padding:28px 40px;margin:0;white-space:pre-wrap\">" + escaped
        + "</pre></body></html>"
    )


def beat(page, seconds: float, note: str = "") -> None:
    """Hold the current view long enough to narrate over it."""
    if note:
        print(f"    {note}  ({seconds * SCALE:.0f}s)")
    page.wait_for_timeout(int(seconds * SCALE * 1000))


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


def convert(webm: Path, out: Path, trim: float = 0.0) -> None:
    """webm -> mp4. yuv420p and even dimensions or Office refuses to play it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(trim), "-i", str(webm),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-movflags", "+faststart", str(out)],
        check=True,
    )


def demo(page, jd_text: str) -> None:
    """The scripted walkthrough. Ordered to match the report's sections.

    Every card is drawn BEFORE the content it names, and the move to that
    content happens underneath it via `then`. An earlier cut navigated first to
    avoid a brief flash of the outgoing page, which meant each section played
    for several seconds before its own title appeared -- far worse than the
    flash it was avoiding.
    """
    card(page, "01", "What this project does",
         "Six AI sub-tasks, one recruitment API", hold=5.0)

    # 1 -- the API surface (requirement 6) ---------------------------------
    print("  [1/12] Swagger")
    # settle: Swagger fetches openapi.json and renders the operation list after
    # "load" fires, so the card has to outlast the load event itself.
    card(page, "02", "The API", "Every sub-task, exposed as an endpoint",
         settle=2.0,
         then=lambda: page.goto(f"{API}/docs", wait_until="commit",
                                timeout=120_000))
    beat(page, 9, "title + endpoint list")
    scroll(page, 900)
    beat(page, 7, "CV and NLP endpoint groups")
    scroll(page, 900)
    beat(page, 7, "orchestration + LLMOps groups")
    scroll(page, -1800, steps=4, pause=0.3)
    beat(page, 3)

    # 2 -- model registry ---------------------------------------------------
    print("  [2/12] Model registry")
    card(page, "03", "Which models, and why",
         "A registry you can query, not a claim in a document",
         then=lambda: json_render(page, "/model-registry"))
    beat(page, 7, "the two CV models")
    # 1000, not 780: the larger type makes the registry a third taller, so the
    # old distance stopped short of the NLP entries this beat narrates.
    scroll(page, 1000, steps=6, pause=0.55)
    beat(page, 8, "the NLP models")

    # 3 -- health -----------------------------------------------------------
    print("  [3/12] Health")
    card(page, "04", "Is it healthy?",
         "Backend reachable, fine-tuned model loaded",
         then=lambda: json_render(page, "/health"))
    beat(page, 11, "backend reachable, fine-tuned model loaded")

    # 4 -- the UI -----------------------------------------------------------
    print("  [4/12] Streamlit home")
    card(page, "05", "The app",
         "One tab per sub-task — the UI holds no model code",
         hold=4.0, settle=7.0,
         then=lambda: page.goto(UI, wait_until="commit", timeout=120_000))
    beat(page, 9, "one tab per sub-task; the client holds no model code")

    # 5 -- ingest -----------------------------------------------------------
    print("  [5/12] Upload + ingest")
    card(page, "06", "Reading the resume",
         "Computer Vision — is it a resume, and what does it say?",
         then=lambda: click_tab(page, 0))
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
    card(page, "07", "Pulling out the facts", "Named entity recognition",
         then=lambda: click_tab(page, 1))
    run_button(page, "Extract entities")
    beat(page, 14, "skills/titles/dates scored; identifiers held back")

    # 7 -- fine-tuned classifier --------------------------------------------
    print("  [7/12] Fit — fine-tuned")
    card(page, "08", "Scoring the fit", "The model we fine-tuned ourselves",
         then=lambda: click_tab(page, 2))
    run_button(page, "Run fine-tuned model")
    beat(page, 14, "label + full score distribution, ~1s on CPU")

    # 8 -- the comparison, the report's centrepiece -------------------------
    print("  [8/12] Fit — both models side by side")
    card(page, "09", "Small model vs large model",
         "The experiment at the centre of this project")
    run_button(page, "Compare both side by side")
    beat(page, 18, "they disagree")
    scroll(page, 500, steps=4)
    beat(page, 8)

    # 9 -- extractive QA -----------------------------------------------------
    print("  [9/12] Question answering")
    card(page, "10", "Asking questions",
         "Answers quoted from the resume — or refused",
         then=lambda: click_tab(page, 3))
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
    card(page, "11", "Writing the brief",
         "Generated from text with personal details removed",
         then=lambda: click_tab(page, 4))
    run_button(page, "Generate brief")
    beat(page, 16, "written from PII-redacted text")
    scroll(page, 700, steps=5)
    beat(page, 12, "strengths, gaps, interview questions")

    # 11 -- the whole chain in one call ---------------------------------------
    print("  [11/12] Full screening")
    card(page, "12", "All of it, in one call", "Six sub-tasks, one request",
         then=lambda: click_tab(page, 5))
    run_button(page, "Screen candidate")
    beat(page, 16, "every sub-task in a single request")
    scroll(page, 800, steps=5)
    beat(page, 12)

    # 12 -- LLMOps ------------------------------------------------------------
    print("  [12/12] LLMOps")
    card(page, "13", "Watching it in production",
         "Seven metrics, measured not asserted",
         then=lambda: click_tab(page, 6))
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
        context.add_init_script(DARK_BG_JS)
        page = context.new_page()
        # Playwright's capture starts before the compositor is at full size, so
        # the opening seconds come out as a half-size frame on a grey field.
        # Record that on a blank screen and cut it off in convert(), rather than
        # spending the first title card on it.
        page.wait_for_timeout(int(WARMUP * 1000))
        globals()["_started"] = started = time.time()
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
    convert(videos[0], out, trim=WARMUP)
    shutil.rmtree(staging, ignore_errors=True)

    # Cue sheet for whoever writes the voice-over: where each card lands.
    cues = out.with_name(out.stem + "_timeline.txt")
    with cues.open("w", encoding="utf-8") as handle:
        for seconds, label in TIMELINE:
            handle.write(f"{int(seconds) // 60}:{int(seconds) % 60:02d}  {label}\n")
        handle.write(f"{int(elapsed) // 60}:{int(elapsed) % 60:02d}  (end)\n")
    print(f"  cue sheet -> {cues.name}")

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  recorded {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    if elapsed < 240:
        print("  under 4 min -- re-run with --scale 1.3 for a slower read")
    elif elapsed > 400:
        print("  over 6.5 min -- re-run with --scale 0.85")


if __name__ == "__main__":
    main()
