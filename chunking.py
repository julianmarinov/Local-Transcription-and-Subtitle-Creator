"""Splits long audio/video files into fixed-length pieces so very long
recordings (1-2hrs) transcribe with a bounded per-chunk decode buffer and
give incremental progress, then stitches the per-chunk results back into
one continuous timeline by offsetting timestamps.

No cross-chunk overlap/dedup - a word split exactly on a chunk boundary is
a rare, minor cosmetic issue, not worth the added complexity here.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backends import Segment, TranscriptionResult

DEFAULT_THRESHOLD_MINUTES = 20.0
DEFAULT_CHUNK_MINUTES = 15.0
MIN_TAIL_SECONDS = 60.0


def probe_duration_seconds(path: str) -> Optional[float]:
    """Duration in seconds, or None when ffprobe can't tell (some streams,
    e.g. raw/live-recorded files, report "N/A") - the caller then simply
    transcribes the file in one go instead of chunking it."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(
            "Could not read this file as audio/video"
            + (f": {detail[-1]}" if detail else ".")
        )
    try:
        duration = float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return None
    return duration if duration > 0 else None


def should_chunk(duration_seconds: float, threshold_minutes: float = DEFAULT_THRESHOLD_MINUTES) -> bool:
    return duration_seconds > threshold_minutes * 60


@dataclass
class Chunk:
    path: str
    start_offset: float


def plan_chunks(duration_seconds: float, chunk_minutes: float = DEFAULT_CHUNK_MINUTES) -> list[tuple[float, float]]:
    """(start, length) pairs covering the whole file. A trailing piece
    shorter than a minute is folded into the previous chunk rather than
    transcribed on its own - a few seconds of audio cut mid-sentence is
    exactly where Whisper produces garbage."""
    chunk_seconds = max(chunk_minutes, 1) * 60
    plan: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        length = min(chunk_seconds, duration_seconds - start)
        plan.append((start, length))
        start += chunk_seconds
    if len(plan) > 1 and plan[-1][1] < MIN_TAIL_SECONDS:
        tail_start, tail_len = plan.pop()
        prev_start, prev_len = plan.pop()
        plan.append((prev_start, prev_len + tail_len))
    return plan


def extract_chunk(input_path: str, start: float, length: float, workdir: str, index: int) -> Chunk:
    """Decodes one piece of the source to 16 kHz mono WAV - the exact
    format Whisper resamples to internally anyway, so this costs nothing
    extra later. -vn skips the video stream entirely for video files."""
    out_path = str(Path(workdir) / f"chunk_{index:03}.wav")
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000", out_path,
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(f"ffmpeg failed to extract audio: {detail[-1] if detail else 'unknown error'}")
    return Chunk(path=out_path, start_offset=start)


def merge_results(chunk_results: list[tuple[Chunk, TranscriptionResult]]) -> TranscriptionResult:
    segments: list[Segment] = []
    language: Optional[str] = None
    for chunk, result in chunk_results:
        if language is None:
            language = result.language
        for seg in result.segments:
            segments.append(Segment(
                start=seg.start + chunk.start_offset,
                end=seg.end + chunk.start_offset,
                text=seg.text,
            ))
    return TranscriptionResult(segments=segments, language=language)
