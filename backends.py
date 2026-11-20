"""Transcription backend abstraction.

Each backend wraps a different Whisper implementation behind a common
interface so the rest of the app doesn't need to know which one is in use.
select_backend() auto-picks the fastest one actually installed:

  1. mlx-whisper   - Apple Silicon only, native Metal GPU. Fastest on Mac.
  2. faster-whisper - CTranslate2. No MPS support (CPU-only on Mac), but
                       faster than openai-whisper on CUDA and CPU.
  3. openai-whisper - reference implementation, always available (required
                       dependency) - the guaranteed fallback.

Loading and transcribing are separate steps (backend.load() -> a
LoadedModel with its own .transcribe()) so a caller processing multiple
chunks of one file only pays the model-load cost once.

torch is imported lazily (only by the backends that actually need it): it
takes seconds to import, and the mlx path on Apple Silicon never uses it,
so importing it at module level only slowed down app startup there.
"""
from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from model_download import DownloadProgressWatcher, ProgressCallback

# Model choices shown in the GUI, in speed order. "turbo" is
# large-v3-turbo: close to large-v3 accuracy at several times the speed,
# a weaker choice only for some lower-resource languages.
MODEL_SIZES = ["tiny", "base", "small", "medium", "turbo", "large"]

# Seconds of audio transcribed so far -> GUI progress. Only backends that
# stream their segments (faster-whisper) can report this.
AudioProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class TranscriptionCancelled(Exception):
    """Raised from inside a backend's transcribe() when the user cancels."""


def _torch():
    import torch
    return torch


def get_device():
    """Best available compute device: CUDA > Apple Silicon MPS > CPU."""
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: Optional[str] = None


class LoadedModel(Protocol):
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        should_cancel: Optional[CancelCheck] = None,
        on_audio_progress: Optional[AudioProgressCallback] = None,
    ) -> TranscriptionResult:
        ...


class TranscriptionBackend(Protocol):
    name: str

    def load(self, model_size: str, on_progress: Optional[ProgressCallback] = None) -> LoadedModel:
        ...

    def device_label(self) -> str:
        ...


def _dict_segments(result: dict) -> list[Segment]:
    return [
        Segment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
        for seg in result["segments"]
        if seg["text"].strip()
    ]


class _OpenAIWhisperLoadedModel:
    def __init__(self, model, device) -> None:
        self._model = model
        self._device = device

    def transcribe(self, audio_path, language=None, should_cancel=None, on_audio_progress=None) -> TranscriptionResult:
        with _torch().no_grad():
            # fp16 autocast is only reliable on CUDA; on MPS it produces
            # garbled/repeated output, and CPU doesn't support it at all.
            # condition_on_previous_text=False: feeding each segment's own
            # previous output back in as decoding context is the standard
            # cause of runaway repetition loops (one bad/repetitive segment
            # poisons every segment after it) - worth the minor loss of
            # cross-segment terminology consistency.
            result = self._model.transcribe(
                audio_path,
                fp16=(self._device.type == "cuda"),
                condition_on_previous_text=False,
                language=language,
            )
        return TranscriptionResult(segments=_dict_segments(result), language=result.get("language"))


class OpenAIWhisperBackend:
    """Reference implementation (openai-whisper). Always available - the
    guaranteed fallback if no faster optional backend is installed."""

    name = "openai-whisper"

    def __init__(self) -> None:
        import whisper
        self._whisper = whisper

    def device_label(self) -> str:
        return get_device().type

    def load(self, model_size: str, on_progress: Optional[ProgressCallback] = None) -> _OpenAIWhisperLoadedModel:
        device = get_device()
        with DownloadProgressWatcher(model_size, on_progress):
            model = self._whisper.load_model(model_size, device=device)
        if device.type == "cuda":
            model = model.half()
        return _OpenAIWhisperLoadedModel(model, device)


def _decode_audio_16k_mono(path: str):
    """Decodes any audio/video file to the 16 kHz mono float32 array Whisper
    works on, using the ffmpeg binary the app already requires/bundles.

    faster-whisper would otherwise decode through PyAV, whose newer releases
    (av >= 16 or so) dropped an argument faster-whisper 1.2.x still passes -
    every file then fails with "open() got an unexpected keyword argument
    'metadata_errors'". Decoding ourselves sidesteps that whole class of
    version breakage and handles exactly the same formats as the rest of
    the app (which all go through ffmpeg too)."""
    import subprocess

    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg could not decode the audio: {detail[-1] if detail else 'unknown error'}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


class _FasterWhisperLoadedModel:
    def __init__(self, model) -> None:
        self._model = model

    def transcribe(self, audio_path, language=None, should_cancel=None, on_audio_progress=None) -> TranscriptionResult:
        # See the matching comment in _OpenAIWhisperLoadedModel.transcribe -
        # condition_on_previous_text=False avoids compounding repetition
        # loops across segments. vad_filter skips silent stretches, which is
        # both faster and where Whisper most often hallucinates text.
        segments_gen, info = self._model.transcribe(
            _decode_audio_16k_mono(audio_path), language=language, condition_on_previous_text=False, vad_filter=True,
        )
        # The generator decodes lazily, segment by segment - so this is the
        # one backend that can be cancelled mid-file and report progress.
        segments: list[Segment] = []
        for seg in segments_gen:
            if should_cancel and should_cancel():
                raise TranscriptionCancelled()
            text = seg.text.strip()
            if text:
                segments.append(Segment(start=seg.start, end=seg.end, text=text))
            if on_audio_progress:
                on_audio_progress(seg.end)
        return TranscriptionResult(segments=segments, language=info.language)


def _cuda_available_without_torch() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class FasterWhisperBackend:
    """CTranslate2-based backend. No MPS support - CUDA or CPU only."""

    name = "faster-whisper"

    def __init__(self) -> None:
        from faster_whisper import WhisperModel
        self._WhisperModel = WhisperModel
        self._device = "cuda" if _cuda_available_without_torch() else "cpu"
        self._compute_type = "float16" if self._device == "cuda" else "int8"

    def device_label(self) -> str:
        return self._device

    def load(self, model_size: str, on_progress: Optional[ProgressCallback] = None) -> _FasterWhisperLoadedModel:
        with DownloadProgressWatcher(model_size, on_progress):
            model = self._WhisperModel(model_size, device=self._device, compute_type=self._compute_type)
        return _FasterWhisperLoadedModel(model)


class _MLXWhisperLoadedModel:
    """mlx_whisper.transcribe() loads (and downloads, if needed) the model
    internally on the first call and caches it for later calls - there's no
    separate load step to hook. The download watcher is therefore attached
    only to the first transcribe() call."""

    def __init__(self, mlx_whisper_module, repo: str, model_size: str, on_progress: Optional[ProgressCallback]) -> None:
        self._mlx_whisper = mlx_whisper_module
        self._repo = repo
        self._model_size = model_size
        self._pending_progress = on_progress

    def transcribe(self, audio_path, language=None, should_cancel=None, on_audio_progress=None) -> TranscriptionResult:
        progress = self._pending_progress
        self._pending_progress = None
        # See the matching comment in _OpenAIWhisperLoadedModel.transcribe.
        with DownloadProgressWatcher(self._model_size, progress):
            result = self._mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=self._repo,
                condition_on_previous_text=False,
                language=language,
                verbose=None,
            )
        return TranscriptionResult(segments=_dict_segments(result), language=result.get("language"))


class MLXWhisperBackend:
    """Apple's MLX framework - native Metal GPU execution. The fastest
    option on Apple Silicon."""

    name = "mlx-whisper"

    _MODEL_REPOS = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "turbo": "mlx-community/whisper-large-v3-turbo",
        "large": "mlx-community/whisper-large-v3-mlx",
    }

    def __init__(self) -> None:
        import mlx_whisper
        self._mlx_whisper = mlx_whisper

    def device_label(self) -> str:
        return "mps (mlx)"

    def load(self, model_size: str, on_progress: Optional[ProgressCallback] = None) -> _MLXWhisperLoadedModel:
        repo = self._MODEL_REPOS.get(model_size, self._MODEL_REPOS["small"])
        return _MLXWhisperLoadedModel(self._mlx_whisper, repo, model_size, on_progress)


_selected_backend: Optional[TranscriptionBackend] = None


def select_backend() -> TranscriptionBackend:
    if _is_apple_silicon() and _importable("mlx_whisper"):
        return MLXWhisperBackend()
    if _importable("faster_whisper"):
        return FasterWhisperBackend()
    return OpenAIWhisperBackend()


def cached_backend() -> TranscriptionBackend:
    """select_backend(), but constructed once per process - backend
    construction imports heavy libraries, so the GUI's Help window and
    every batch run shouldn't each pay that again."""
    global _selected_backend
    if _selected_backend is None:
        _selected_backend = select_backend()
    return _selected_backend
