"""Orchestrates transcription runs (single file or a batch/queue). No GUI
dependency - errors are reported through callbacks so this stays
independently testable."""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Optional

import chunking
import remote_input
from backends import (
    LoadedModel,
    Segment,
    TranscriptionBackend,
    TranscriptionCancelled,
    TranscriptionResult,
    cached_backend,
)
from subtitles import build_json, build_srt, build_txt, build_vtt, write_text_file

StatusCallback = Callable[[str], None]
TextCallback = Callable[[str], None]
ErrorCallback = Callable[[str, str], None]  # (message, full_traceback)
ItemStatusCallback = Callable[[str, str], None]  # (input_path, status)
ItemDoneCallback = Callable[[str, str], None]  # (input_path, outcome)
FinishedCallback = Callable[[str], None]  # final summary message

# Per-item outcomes reported to on_item_complete.
OUTCOME_DONE = "done"
OUTCOME_ERROR = "error"
OUTCOME_CANCELLED = "cancelled"

_FORMAT_BUILDERS: dict[str, Callable[[list[Segment], Optional[str]], str]] = {
    "srt": lambda segments, language: build_srt(segments),
    "vtt": lambda segments, language: build_vtt(segments),
    "txt": lambda segments, language: build_txt(segments),
    "json": build_json,
}

AUDIO_VIDEO_EXTENSIONS = (
    "*.mp3", "*.wav", "*.m4a", "*.aac", "*.flac", "*.ogg", "*.opus", "*.wma",
    "*.mp4", "*.mov", "*.mkv", "*.webm", "*.avi", "*.m4v",
)

# Windows refuses these as file names regardless of extension.
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _is_oom_error(exc: Exception) -> bool:
    if exc.__class__.__name__ in ("OutOfMemoryError", "MemoryError"):
        return True
    text = str(exc).lower()
    return "out of memory" in text or " oom" in text


def _sanitize_filename(name: str, max_len: int = 150) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(". ")
    name = name[:max_len].rstrip(". ")
    if name.upper() in _WINDOWS_RESERVED:
        name = f"_{name}"
    return name or "youtube_audio"


def _unique_output_bases(input_paths: list[str]) -> dict[str, Optional[str]]:
    """Local files in the same folder that differ only by extension
    (talk.mp3 + talk.mp4) would otherwise both write talk.srt, the second
    silently overwriting the first. Those get the source extension kept in
    the name instead (talk.mp3.srt, talk.mp4.srt). URLs resolve their name
    only after download, so they're returned as None here."""
    bases: dict[str, Optional[str]] = {}
    seen: dict[str, int] = {}
    for p in input_paths:
        if remote_input.is_url(p):
            bases[p] = None
            continue
        base = os.path.splitext(p)[0]
        key = os.path.normcase(base)
        seen[key] = seen.get(key, 0) + 1
        bases[p] = base
    for p, base in bases.items():
        if base is not None and seen[os.path.normcase(base)] > 1:
            bases[p] = p  # keep ".mp3" as part of the base
    return bases


def _unique_path_for(base: str, taken: set[str]) -> str:
    candidate, n = base, 2
    while os.path.normcase(candidate) in taken:
        candidate = f"{base} ({n})"
        n += 1
    taken.add(os.path.normcase(candidate))
    return candidate


def _fmt_clock(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


class TranscriptionService:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel_transcription(self) -> None:
        self.cancelled = True

    def _should_cancel(self) -> bool:
        return self.cancelled

    def transcribe(
        self,
        input_path: str,
        model_type: str,
        formats: set[str],
        update_status: StatusCallback,
        update_transcription_text: TextCallback,
        on_error: Optional[ErrorCallback] = None,
        chunk_minutes: float = chunking.DEFAULT_CHUNK_MINUTES,
        output_base_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe a single file or URL. input_path may be a local path
        or a YouTube (or other yt-dlp supported) URL. output_base_path is
        auto-derived if not given: next to the source for local files, or
        into ~/Downloads named after the video's title for a URL. Returns
        the outcome (OUTCOME_*)."""
        self.cancelled = False
        model = self._load_model(model_type, update_status, on_error)
        if model is None:
            return OUTCOME_ERROR
        return self._process_file(
            model, input_path, model_type, formats,
            update_status, update_transcription_text, on_error, chunk_minutes, output_base_path, language,
        )

    def transcribe_batch(
        self,
        input_paths: list[str],
        model_type: str,
        formats: set[str],
        update_status: StatusCallback,
        update_transcription_text: TextCallback,
        on_error: Optional[ErrorCallback] = None,
        on_item_status: Optional[ItemStatusCallback] = None,
        on_item_complete: Optional[ItemDoneCallback] = None,
        chunk_minutes: float = chunking.DEFAULT_CHUNK_MINUTES,
        language: Optional[str] = None,
        on_finished: Optional[FinishedCallback] = None,
    ) -> None:
        """Transcribe a queue of files and/or URLs, writing each output
        next to its source file (or into ~/Downloads for a URL, named
        after the video's title). The model is loaded once and reused
        across all items. One item's failure doesn't stop the rest of
        the batch.

        on_finished is ALWAYS called exactly once when the run ends -
        complete, cancelled, or failed - so a GUI can rely on it to reset
        itself instead of pattern-matching status strings."""
        self.cancelled = False
        summary = "Nothing to do."
        try:
            if not input_paths:
                return

            model = self._load_model(model_type, update_status, on_error)
            if model is None:
                summary = "Error: could not load the model."
                return

            bases = _unique_output_bases(input_paths)
            taken: set[str] = set()
            counts = {OUTCOME_DONE: 0, OUTCOME_ERROR: 0, OUTCOME_CANCELLED: 0}
            total = len(input_paths)
            for i, input_path in enumerate(input_paths, start=1):
                if self.cancelled:
                    break

                name = input_path if remote_input.is_url(input_path) else os.path.basename(input_path)
                update_status(f"Processing {i}/{total}: {name}")
                if on_item_status:
                    item_status: StatusCallback = lambda msg, path=input_path: on_item_status(path, msg)
                else:
                    item_status = lambda msg, i=i, name=name: update_status(f"[{i}/{total}] {name}: {msg}")

                base = bases[input_path]
                if base is not None:
                    base = _unique_path_for(base, taken)

                outcome = self._process_file(
                    model, input_path, model_type, formats,
                    item_status, update_transcription_text, on_error, chunk_minutes, base, language,
                    taken_bases=taken,
                )
                counts[outcome] += 1
                if on_item_complete:
                    on_item_complete(input_path, outcome)

            if self.cancelled:
                summary = f"Cancelled. {counts[OUTCOME_DONE]} of {total} file(s) finished."
            elif counts[OUTCOME_ERROR]:
                summary = f"Finished with errors: {counts[OUTCOME_DONE]} done, {counts[OUTCOME_ERROR]} failed."
            else:
                summary = f"Batch complete: {total} file(s) processed."
            update_status(summary)
        finally:
            if on_finished:
                on_finished(summary)

    def _load_model(
        self, model_type: str, update_status: StatusCallback, on_error: Optional[ErrorCallback]
    ) -> Optional[LoadedModel]:
        try:
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")

            backend: TranscriptionBackend = cached_backend()
            update_status(f"Loading {backend.name} '{model_type}' model on {backend.device_label()}...")

            def on_progress(percent: int, mb_done: float, mb_total: float) -> None:
                update_status(f"Downloading model... {percent}% ({mb_done:.0f}MB/{mb_total:.0f}MB)")

            try:
                return backend.load(model_type, on_progress=on_progress)
            except Exception as e:
                if _is_oom_error(e):
                    raise RuntimeError(
                        f"Ran out of memory loading the '{model_type}' model. "
                        "Try a smaller model size (e.g. 'turbo' or 'small')."
                    ) from None
                raise
        except Exception as e:
            tb = traceback.format_exc()
            update_status("Error loading the model.")
            if on_error:
                on_error(str(e), tb)
            return None

    def _process_file(
        self,
        model: LoadedModel,
        input_path: str,
        model_type: str,
        formats: set[str],
        update_status: StatusCallback,
        update_transcription_text: TextCallback,
        on_error: Optional[ErrorCallback],
        chunk_minutes: float,
        output_base_path: Optional[str] = None,
        language: Optional[str] = None,
        taken_bases: Optional[set[str]] = None,
    ) -> str:
        try:
            if not formats:
                raise ValueError("Select at least one output format.")

            with contextlib.ExitStack() as stack:
                actual_path = input_path
                resolved_output_base = output_base_path

                if remote_input.is_url(input_path):
                    update_status("Downloading audio from URL...")
                    tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="audio_to_srt_yt_"))
                    actual_path, title = remote_input.download_audio(input_path, tmp_dir)
                    if resolved_output_base is None:
                        downloads_dir = Path.home() / "Downloads"
                        downloads_dir.mkdir(parents=True, exist_ok=True)
                        resolved_output_base = str(downloads_dir / _sanitize_filename(title))
                        if taken_bases is not None:
                            resolved_output_base = _unique_path_for(resolved_output_base, taken_bases)
                elif not os.path.isfile(input_path):
                    raise FileNotFoundError(f"File not found: {input_path}")
                elif resolved_output_base is None:
                    resolved_output_base = os.path.splitext(input_path)[0]

                result = self._run_transcription(
                    actual_path, model, model_type, update_status, chunk_minutes, language,
                )
                if result is None:  # cancelled
                    update_status("Cancelled")
                    return OUTCOME_CANCELLED

                if not result.segments:
                    update_status("Done - no speech detected")
                else:
                    update_status("Writing files...")
                for fmt in sorted(formats):
                    content = _FORMAT_BUILDERS[fmt](result.segments, result.language)
                    # Plain append, not with_extension(): the base never has a
                    # media extension at this point, but may legitimately
                    # contain dots ("Episode 1.5" from a video title, or the
                    # kept "talk.mp3" from _unique_output_bases).
                    write_text_file(content, f"{resolved_output_base}.{fmt}")

                full_text = "\n".join(seg.text for seg in result.segments)
                update_transcription_text(full_text)
                if result.segments:
                    lang = f" [{result.language}]" if result.language else ""
                    update_status(f"Done{lang}")
                return OUTCOME_DONE
        except TranscriptionCancelled:
            update_status("Cancelled")
            return OUTCOME_CANCELLED
        except Exception as e:
            tb = traceback.format_exc()
            update_status(f"Error: {e}")
            if on_error:
                on_error(f"{os.path.basename(input_path) or input_path}: {e}", tb)
            return OUTCOME_ERROR

    def _run_transcription(
        self,
        input_path: str,
        model: LoadedModel,
        model_type: str,
        update_status: StatusCallback,
        chunk_minutes: float,
        language: Optional[str] = None,
    ) -> Optional[TranscriptionResult]:
        duration = chunking.probe_duration_seconds(input_path)
        if duration is None or not chunking.should_chunk(duration):
            update_status("Transcribing...")
            return self._transcribe_one(
                model, input_path, model_type, language, self._progress_reporter(update_status, 0.0, duration),
            )

        with tempfile.TemporaryDirectory(prefix="audio_to_srt_chunks_") as workdir:
            chunk_results = []
            # Chunks are extracted lazily, one ahead of transcription, so
            # cancelling doesn't first wait for ffmpeg to split the whole
            # file and the temp dir never holds more than one chunk.
            chunk_plan = chunking.plan_chunks(duration, chunk_minutes)
            for i, (start, length) in enumerate(chunk_plan, start=1):
                if self.cancelled:
                    return None
                update_status(f"Transcribing part {i}/{len(chunk_plan)}...")
                chunk = chunking.extract_chunk(input_path, start, length, workdir, i)
                try:
                    result = self._transcribe_one(
                        model, chunk.path, model_type, language,
                        self._progress_reporter(update_status, start, duration, f"part {i}/{len(chunk_plan)}"),
                    )
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(chunk.path)
                chunk_results.append((chunk, result))
                # Lock the language after the first chunk: re-detecting it
                # per chunk occasionally flips to a wrong language on a
                # chunk that starts with music or silence.
                if language is None and result.language:
                    language = result.language
            return chunking.merge_results(chunk_results)

    @staticmethod
    def _progress_reporter(update_status: StatusCallback, offset: float, total: Optional[float], label: str = ""):
        if not total:
            return None
        prefix = f"Transcribing {label} " if label else "Transcribing "

        def report(seconds_done: float) -> None:
            done = min(offset + seconds_done, total)
            update_status(f"{prefix}{int(done / total * 100)}% ({_fmt_clock(done)} / {_fmt_clock(total)})")
        return report

    def _transcribe_one(
        self,
        model: LoadedModel,
        audio_path: str,
        model_type: str,
        language: Optional[str] = None,
        on_audio_progress=None,
    ) -> TranscriptionResult:
        try:
            return model.transcribe(
                audio_path, language=language, should_cancel=self._should_cancel, on_audio_progress=on_audio_progress,
            )
        except TranscriptionCancelled:
            raise
        except Exception as e:
            if _is_oom_error(e):
                raise RuntimeError(
                    f"Ran out of memory while transcribing with the '{model_type}' model. "
                    "Try a smaller model size (e.g. 'turbo' or 'small')."
                ) from None
            raise
