"""
Prepend a title card carrying the student's name and ID to the demo video.

    python scripts/add_title_card.py --id 2024AC05266 --name "Aastha Sakshi"

Reads docs/demo.mp4 and writes docs/demo_titled.mp4.

Kept separate from record_demo.py, and its output kept out of git, for the same
reason REPORT.md carries {{BITS_ID}} placeholders: the repository is public, so
the committed video is the one with no personal details on it, and the titled
copy that gets submitted stays local. Re-running the recorder does not silently
republish a name.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT = 1920, 1080

TITLE = """
<html><body style="margin:0;width:1920px;height:1080px;display:flex;
 align-items:center;justify-content:center;background:#0f1420;
 font-family:'Segoe UI',system-ui,sans-serif;color:#e8ecf4">
 <div style="max-width:1420px;padding:0 100px">
  <div style="font:600 27px/1 'Segoe UI';letter-spacing:.34em;color:#5b8def">
   {course}</div>
  <div style="height:3px;width:96px;background:#5b8def;margin:30px 0 40px"></div>
  <div style="font:600 92px/1.1 'Segoe UI';letter-spacing:-.02em">{title}</div>
  <div style="font:300 35px/1.5 'Segoe UI';color:#93a2bd;margin-top:28px">
   {subtitle}</div>
  <div style="margin-top:76px;display:flex;gap:78px;
   font:400 28px/1.6 'Segoe UI';color:#c7d2e6">
   {who}
  </div>
 </div></body></html>
"""

FIELD = ("<div><div style=\"font:600 15px/1 'Segoe UI';letter-spacing:.22em;"
         "color:#6b7a99;margin-bottom:11px\">{label}</div>{value}</div>")


def render(png: Path, student_id: str, name: str) -> None:
    who = ""
    if name:
        who += FIELD.format(label="SUBMITTED BY", value=name)
    if student_id:
        who += FIELD.format(label="BITS ID", value=student_id)
    who += FIELD.format(label="DOMAIN", value="HR / Recruitment")
    who += FIELD.format(label="CATEGORIES", value="Computer Vision + NLP")

    html = TITLE.format(
        course="CCZG506 · ASSIGNMENT II",
        title="AI Recruitment Assistant",
        subtitle="Screening one candidate against one job description, "
                 "end to end, over an API.",
        who=who,
    )
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        page = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT}).new_page()
        page.set_content(html)
        page.wait_for_timeout(700)  # let the webfont settle before capturing
        page.screenshot(path=str(png))
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepend a title card")
    parser.add_argument("--in", dest="src", default=str(ROOT / "docs" / "demo.mp4"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo_titled.mp4"))
    parser.add_argument("--id", dest="student_id", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="how long the title card holds")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"missing {src} -- run scripts/record_demo.py first")

    staging = Path(tempfile.mkdtemp(prefix="title_"))
    png = staging / "title.png"
    render(png, args.student_id, args.name)
    print(f"  rendered title card ({png.stat().st_size // 1024} KB)")

    # One concat pass rather than two encodes: the still is turned into frames
    # and joined to the demo in the same graph. Both sides are forced to the
    # same fps/size/pixel format first, or concat refuses them.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-loop", "1", "-t", str(args.seconds), "-i", str(png),
         "-i", str(src),
         "-filter_complex",
         f"[0:v]scale={WIDTH}:{HEIGHT},fps=25,format=yuv420p,setsar=1[t];"
         f"[1:v]scale={WIDTH}:{HEIGHT},fps=25,format=yuv420p,setsar=1[v];"
         "[t][v]concat=n=2:v=1:a=0[out]",
         "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        check=True,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  title card holds {args.seconds:.0f}s — every later cue shifts by that much")


if __name__ == "__main__":
    main()
