"""Pure subtitle-formatting logic - no GUI or backend dependencies, easy to
unit test."""
from __future__ import annotations

import json
import os
import textwrap

from backends import Segment

# Standard subtitle readability convention (~42 chars/line, max 2 lines
# on screen at once).
DEFAULT_LINE_WIDTH = 42
DEFAULT_MAX_LINES = 2


def fmt_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hrs, rem = divmod(total_ms, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"


def fmt_vtt_time(seconds: float) -> str:
    return fmt_srt_time(seconds).replace(",", ".")


def wrap_subtitle_text(text: str, max_chars: int = DEFAULT_LINE_WIDTH) -> str:
    text = text.strip()
    if not text:
        return text
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False))


def split_long_segments(
    segments: list[Segment], max_chars: int = DEFAULT_LINE_WIDTH, max_lines: int = DEFAULT_MAX_LINES,
) -> list[Segment]:
    """Whisper segments can run 20-30 seconds, which wraps to 4-6 lines -
    unreadable as a single subtitle, and most players cut it off. Splits
    any segment that wraps to more than max_lines into consecutive cues of
    at most max_lines each, dividing the segment's time span between them
    in proportion to their text length."""
    out: list[Segment] = []
    for seg in segments:
        lines = textwrap.wrap(seg.text.strip(), width=max_chars, break_long_words=False)
        if len(lines) <= max_lines:
            out.append(seg)
            continue
        groups = [" ".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]
        total_chars = sum(len(g) for g in groups)
        span = max(seg.end - seg.start, 0.0)
        t = seg.start
        for j, g in enumerate(groups):
            end = seg.end if j == len(groups) - 1 else t + span * len(g) / total_chars
            out.append(Segment(start=t, end=end, text=g))
            t = end
    return out


def build_srt(segments: list[Segment]) -> str:
    parts = []
    for i, seg in enumerate(split_long_segments(segments), start=1):
        start = fmt_srt_time(seg.start)
        end = fmt_srt_time(seg.end)
        parts.append(f"{i}\n{start} --> {end}\n{wrap_subtitle_text(seg.text)}\n\n")
    return "".join(parts)


def build_vtt(segments: list[Segment]) -> str:
    parts = ["WEBVTT\n\n"]
    for i, seg in enumerate(split_long_segments(segments), start=1):
        start = fmt_vtt_time(seg.start)
        end = fmt_vtt_time(seg.end)
        parts.append(f"{i}\n{start} --> {end}\n{wrap_subtitle_text(seg.text)}\n\n")
    return "".join(parts)


def build_txt(segments: list[Segment]) -> str:
    """Plain transcript, no timestamps."""
    return "\n".join(seg.text.strip() for seg in segments) + "\n"


def build_json(segments: list[Segment], language: str | None) -> str:
    payload = {
        "language": language,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def with_extension(path: str, ext: str) -> str:
    base, _ = os.path.splitext(path)
    return f"{base}.{ext}"


def write_text_file(content: str, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
