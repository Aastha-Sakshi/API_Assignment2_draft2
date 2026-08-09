"""
Record a voice-over for one of the demo videos, and mux it on.

    python scripts/record_voiceover.py --list
    python scripts/record_voiceover.py --video docs/demo_swagger.mp4
    python scripts/record_voiceover.py --video docs/demo.mp4 --audio take2.wav

The obvious way to do this is to play the video and talk over it, but then the
recording only lines up if you hit play and record at the same instant, and it
drifts if you pause. Instead this records against the clock: it counts you in,
then prints each cue as its moment arrives and stops at the video's exact
duration. The audio timeline and the video timeline are the same by
construction, so muxing is a straight overlay with nothing to nudge.

Cues come from the video's own timeline file if the recorder wrote one --
docs/demo.mp4 -> docs/demo_timeline.txt -- and fall back to a plain clock.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def devices() -> list[str]:
    """Every DirectShow audio input ffmpeg can see."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow",
         "-i", "dummy"],
        capture_output=True, text=True,
    ).stderr
    return re.findall(r'"([^"]+)" \(audio\)', out)


def duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


_ASCII = str.maketrans({"—": " - ", "–": "-", "’": "'", "‘": "'",
                        "“": '"', "”": '"', "…": "..."})


def speakable(text: str) -> str:
    """Text the console can actually print.

    The Windows console is cp1252 by default, so an em dash in a line you are
    about to read aloud arrives as a replacement box. Fold to ASCII only when
    the terminal cannot take the real thing.
    """
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return text
    except UnicodeEncodeError:
        return text.translate(_ASCII).encode("ascii", "ignore").decode()


def read_marks(path: Path) -> list[tuple[float, str]]:
    """Parse `M:SS  text` lines, ignoring blanks and # comments."""
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"(\d+):(\d\d)\s+(.*)", line)
        if match:
            seconds = int(match.group(1)) * 60 + int(match.group(2))
            found.append((float(seconds), match.group(3)))
    return found


def cues(video: Path, total: float) -> list[tuple[float, str]]:
    """(seconds, text) pairs to print while recording.

    Preference order: the words to actually speak, then the recorder's section
    cue sheet, then a bare clock. A written line beats a section title, because
    at 0:41 you want to read something, not be told where you are.
    """
    lines = video.with_name(video.stem + "_lines.txt")
    if lines.exists():
        found = read_marks(lines)
        if found:
            return sorted(found)

    timeline = video.with_name(video.stem + "_timeline.txt")
    found = read_marks(timeline) if timeline.exists() else []
    # A clip with no title cards has a timeline holding only "(end)". That is
    # not enough to keep your place by, so fall back to a plain clock.
    if len([c for c in found if "(end)" not in c[1]]) < 2:
        found = [(float(s), "") for s in range(0, int(total), 15)]
    return sorted(found)


def record(device: str, seconds: float, wav: Path, marks) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    for count in (3, 2, 1):
        print(f"  {count}...", flush=True)
        time.sleep(1)

    process = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "dshow", "-i", f"audio={device}",
         # A second of tail: better to trim silence than to clip your last word.
         "-t", str(seconds + 1.0), "-ac", "1", "-ar", "48000", str(wav)],
        stdin=subprocess.DEVNULL,
    )

    print("\n  SPEAK NOW\n", flush=True)
    started = time.time()
    pending = list(marks)
    while True:
        elapsed = time.time() - started
        if elapsed >= seconds:
            break
        while pending and pending[0][0] <= elapsed:
            at, label = pending.pop(0)
            stamp = f"{int(at) // 60}:{int(at) % 60:02d}"
            print(f"  {stamp}  {speakable(label)}" if label else f"  {stamp}",
                  flush=True)
        time.sleep(0.05)

    print(f"\n  done ({seconds:.0f}s) -- finishing the file", flush=True)
    process.wait(timeout=30)
    if process.returncode != 0:
        raise SystemExit(f"ffmpeg exited {process.returncode} -- no audio captured")


def mux(video: Path, wav: Path, out: Path) -> None:
    """Overlay the voice track. The video stream is copied, not re-encoded."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(video), "-i", str(wav),
         "-map", "0:v:0", "-map", "1:a:0",
         # Stereo 44.1k rather than the mono 48k that came off the mic: some
         # players and upload pipelines are picky about mono AAC, and matching
         # the most ordinary format going costs nothing here.
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-ac", "2", "-ar", "44100",
         # apad before -shortest, or a voice track that came in short would
         # truncate the picture to match it. Padding with silence first means
         # -shortest can only ever cut the audio, so the video stays whole
         # whichever way the lengths fall.
         "-af", "apad", "-shortest",
         "-movflags", "+faststart", str(out)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a voice-over")
    parser.add_argument("--video", default=str(ROOT / "docs" / "demo.mp4"))
    parser.add_argument("--out", help="default: <video>_vo.mp4")
    parser.add_argument("--device", help="microphone name; --list to see them")
    parser.add_argument("--audio", help="mux an existing .wav instead of recording")
    parser.add_argument("--list", action="store_true", help="list microphones and exit")
    parser.add_argument("--script", action="store_true",
                        help="print the lines with their timings and exit "
                             "(rehearse without recording)")
    args = parser.parse_args()

    if args.list:
        for name in devices():
            print(f"  {name}")
        return

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"missing {video}")
    out = Path(args.out) if args.out else video.with_name(video.stem + "_vo.mp4")
    total = duration(video)

    if args.script:
        print(f"\n  {video.name} runs {int(total) // 60}:{int(total) % 60:02d}\n")
        for at, text in cues(video, total):
            print(f"  {int(at) // 60}:{int(at) % 60:02d}  {speakable(text)}")
        return

    if args.audio:
        wav = Path(args.audio)
        if not wav.exists():
            raise SystemExit(f"missing {wav}")
    else:
        available = devices()
        if not available:
            raise SystemExit("ffmpeg found no audio input devices")
        device = args.device or available[0]
        if device not in available:
            raise SystemExit(f"no such microphone: {device!r}\n  have: {available}")
        wav = video.with_name(video.stem + "_vo.wav")

        print(f"\n  {video.name} runs {int(total) // 60}:{int(total) % 60:02d}")
        print(f"  recording from: {device}")
        marks = cues(video, total)
        source = video.with_name(video.stem + "_lines.txt")
        print(f"  script: {source.name}" if source.exists()
              else "  no lines file -- printing a bare clock")
        print("  read each line as it appears; the clock is the video's\n")
        record(device, total, wav, marks)

    mux(video, wav, out)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.audio:
        print(f"  voice track kept at {wav.name} -- remux with --audio {wav.name}")


if __name__ == "__main__":
    main()
