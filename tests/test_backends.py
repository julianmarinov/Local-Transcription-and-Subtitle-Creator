import pytest

import backends

torch = pytest.importorskip("torch")


def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert backends.get_device().type == "cuda"


def test_get_device_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert backends.get_device().type == "mps"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    assert backends.get_device().type == "cpu"


class _StubBackend:
    def __init__(self, name):
        self.name = name


def test_select_backend_prefers_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(backends, "_importable", lambda name: True)
    monkeypatch.setattr(backends, "MLXWhisperBackend", lambda: _StubBackend("mlx-whisper"))
    monkeypatch.setattr(backends, "FasterWhisperBackend", lambda: _StubBackend("faster-whisper"))
    assert backends.select_backend().name == "mlx-whisper"


def test_select_backend_skips_mlx_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(backends, "_importable", lambda name: True)
    monkeypatch.setattr(backends, "FasterWhisperBackend", lambda: _StubBackend("faster-whisper"))
    assert backends.select_backend().name == "faster-whisper"


def test_select_backend_falls_back_to_openai_whisper(monkeypatch):
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(backends, "_importable", lambda name: False)
    monkeypatch.setattr(backends, "OpenAIWhisperBackend", lambda: _StubBackend("openai-whisper"))
    assert backends.select_backend().name == "openai-whisper"


def test_select_backend_faster_whisper_only_if_mlx_missing(monkeypatch):
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(backends, "_importable", lambda name: name == "faster_whisper")
    monkeypatch.setattr(backends, "FasterWhisperBackend", lambda: _StubBackend("faster-whisper"))
    assert backends.select_backend().name == "faster-whisper"


def test_turbo_is_offered_and_mapped_for_mlx():
    assert "turbo" in backends.MODEL_SIZES
    assert set(backends.MODEL_SIZES) <= set(backends.MLXWhisperBackend._MODEL_REPOS)


def test_backends_module_does_not_import_torch_eagerly():
    import subprocess, sys
    code = "import sys, backends; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
