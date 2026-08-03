"""
Capture the report screenshots from the running services.

    python scripts/capture_screenshots.py

Requires the API on :8000 and Streamlit on :8501, and Microsoft Edge installed
(Playwright drives the system browser via channel="msedge" -- no separate
browser download).

Writes PNGs to docs/screenshots/ with names matching the report sections, so
each one can be dropped straight into the Word document.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
API = "http://127.0.0.1:8000"
UI = "http://127.0.0.1:8501"

# Overridden with --resume. Defaults to the synthetic sample rather than a
# real CV: the populated screenshots render the contact block in full, so a
# real resume here means every capture carries that person's personal data.
RESUME = ROOT / "data" / "sample_resume.txt"


def wait_for(url: str, label: str, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            print(f"  {label} ready")
            return True
        except Exception:
            time.sleep(3)
    print(f"  WARNING: {label} not reachable at {url} — skipping its shots")
    return False


def shoot(page, path: Path, full: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=full)
    print(f"  wrote {path.relative_to(ROOT)}")


def json_page(page, endpoint: str, out: Path) -> None:
    """Render a JSON endpoint as readable, monospaced, syntax-neutral HTML.

    The browser's raw JSON view is a single unwrapped line at small type, which
    screenshots badly. Pretty-printing it into a styled <pre> keeps the content
    identical while making the image legible in a printed report.
    """
    raw = urllib.request.urlopen(API + endpoint, timeout=120).read().decode()
    pretty = json.dumps(json.loads(raw), indent=2)
    page.set_content(
        "<html><body style='margin:0;background:#1e1e1e'>"
        f"<div style='background:#0d0d0d;color:#7fd7ff;font:600 15px Consolas,monospace;"
        f"padding:10px 16px'>GET {endpoint}</div>"
        "<pre style='color:#d4d4d4;font:13px Consolas,monospace;padding:16px;"
        f"margin:0;white-space:pre-wrap'>{pretty}</pre></body></html>"
    )
    shoot(page, out)


def tab(page, index: int):
    """Streamlit tabs are buttons; clicking re-renders the panel below them."""
    page.locator('button[role="tab"]').nth(index).click()
    page.wait_for_timeout(1500)


def run_button(page, label: str, wait_ms: int = 90_000) -> None:
    """
    Click a button and wait for Streamlit to finish the round trip.

    Streamlit shows a "RUNNING" status widget while a callback is in flight;
    waiting for it to disappear is more reliable than a fixed sleep, since these
    calls range from 0.5s (local classifier) to ~60s (the reasoning model).
    """
    button = page.get_by_role("button", name=label, exact=False).first
    button.scroll_into_view_if_needed()
    button.click()
    page.wait_for_timeout(1200)
    deadline = time.time() + wait_ms / 1000
    while time.time() < deadline:
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        page.wait_for_timeout(1000)
    page.wait_for_timeout(2500)


def drive_streamlit(page) -> None:
    """
    Populate the UI before capturing it. Screenshots of empty widgets prove the
    app renders, not that it works -- the report needs the latter.
    """
    page.goto(UI, wait_until="networkidle")
    page.wait_for_timeout(7000)
    shoot(page, OUT / "10_streamlit_home.png")

    # Tab 1 -- upload the real resume. Selecting the file is not enough: the app
    # only calls the API when "Ingest document" is pressed, and every later tab
    # is disabled until that populates st.session_state.resume_text.
    tab(page, 0)
    page.locator('input[type="file"]').set_input_files(str(RESUME))
    page.wait_for_timeout(4000)
    run_button(page, "Ingest document", 180_000)
    shoot(page, OUT / "11_streamlit_upload_extracted.png")

    # The job description feeds fit classification, the brief and full screening.
    jd = (ROOT / "data" / "sample_jd.txt").read_text(encoding="utf-8")
    jd_box = page.get_by_placeholder("Paste the JD here", exact=False)
    if jd_box.count():
        jd_box.first.fill(jd)
        jd_box.first.blur()
        page.wait_for_timeout(3000)
        print("  job description filled")
    else:
        print("  WARNING: job-description box not found")

    steps = [
        (1, "Extract entities", "12_streamlit_entities.png", 90_000),
        (2, "Run fine-tuned model", "13_streamlit_fit_finetuned.png", 90_000),
        (2, "Compare both side by side", "14_streamlit_fit_compare.png", 180_000),
        (3, "Ask", "15_streamlit_qa.png", 120_000),
        (4, "Generate brief", "16_streamlit_brief.png", 240_000),
        (5, "Screen candidate", "17_streamlit_full_screening.png", 300_000),
    ]
    for index, label, name, budget in steps:
        tab(page, index)
        try:
            run_button(page, label, budget)
        except Exception as exc:
            print(f"  (tab {index} '{label}' failed: {str(exc)[:80]})")
        shoot(page, OUT / name)

    # LLMOps tab last, so it reflects every call the session just made.
    tab(page, 6)
    try:
        run_button(page, "Refresh /metrics", 60_000)
    except Exception as exc:
        print(f"  (metrics refresh failed: {str(exc)[:80]})")
    shoot(page, OUT / "18_streamlit_llmops.png")


def main():
    global RESUME

    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="msedge", help="msedge | chrome | chromium")
    parser.add_argument("--resume", default=str(RESUME),
                        help="resume to drive the UI with (PDF/PNG/TXT)")
    args = parser.parse_args()

    RESUME = Path(args.resume)
    if not RESUME.exists():
        raise SystemExit(f"resume not found: {RESUME}")
    print(f"driving the UI with {RESUME.name}")

    OUT.mkdir(parents=True, exist_ok=True)
    api_up = wait_for(API + "/health", "API")
    ui_up = wait_for(UI, "Streamlit")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})

        if api_up:
            print("\nAPI:")
            page.goto(API + "/docs", wait_until="networkidle")
            page.wait_for_timeout(2500)
            shoot(page, OUT / "01_swagger_endpoints.png")

            # Expand one endpoint so the schema is visible, not just the list.
            try:
                page.click("#operations-NLP_4___Fit_classification-classify_fit_classify_fit_post")
                page.wait_for_timeout(1200)
                shoot(page, OUT / "02_swagger_classify_fit.png")
            except Exception as exc:
                print(f"  (could not expand /classify-fit: {str(exc)[:70]})")

            for endpoint, name in [
                ("/model-registry", "03_model_registry.png"),
                ("/health", "04_health.png"),
                ("/metrics", "05_metrics_llmops.png"),
            ]:
                json_page(page, endpoint, OUT / name)

        if ui_up:
            print("\nStreamlit:")
            drive_streamlit(page)

        browser.close()

    written = sorted(OUT.glob("*.png"))
    print(f"\n{len(written)} screenshots in {OUT}")
    for path in written:
        print(f"  {path.name}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
