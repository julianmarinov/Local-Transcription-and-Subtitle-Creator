from transcription_service import _is_oom_error, _sanitize_filename


def test_is_oom_error_by_message():
    assert _is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert _is_oom_error(RuntimeError("MPS backend out of memory (MPS allocated: ...)"))
    assert not _is_oom_error(ValueError("unrelated error"))


def test_is_oom_error_by_class_name():
    class OutOfMemoryError(Exception):
        pass

    assert _is_oom_error(OutOfMemoryError("generic"))


def test_sanitize_filename_replaces_illegal_chars():
    assert _sanitize_filename('a/b:c*d?"e') == "a_b_c_d__e"


def test_sanitize_filename_empty_falls_back():
    assert _sanitize_filename("   ") == "youtube_audio"


def test_sanitize_filename_windows_reserved_and_trailing_dots():
    assert _sanitize_filename("CON") == "_CON"
    assert _sanitize_filename("title...") == "title"
    assert len(_sanitize_filename("x" * 500)) <= 150


def test_same_stem_different_extension_get_distinct_outputs(tmp_path):
    from transcription_service import _unique_output_bases

    a, b, c = str(tmp_path / "talk.mp3"), str(tmp_path / "talk.mp4"), str(tmp_path / "other.wav")
    bases = _unique_output_bases([a, b, c, "https://example.com/v"])
    assert bases[a] != bases[b]
    assert bases[c] == str(tmp_path / "other")
    assert bases["https://example.com/v"] is None


# --- batch behaviour with a fake model (no ffmpeg/whisper needed) ---

import transcription_service as ts
from backends import Segment, TranscriptionCancelled, TranscriptionResult


class _FakeModel:
    def __init__(self, service=None, cancel_on_call=None, fail_on=None):
        self.calls = 0
        self.service = service
        self.cancel_on_call = cancel_on_call
        self.fail_on = fail_on or set()

    def transcribe(self, audio_path, language=None, should_cancel=None, on_audio_progress=None):
        self.calls += 1
        if self.cancel_on_call == self.calls:
            self.service.cancel_transcription()
            raise TranscriptionCancelled()
        if any(audio_path.endswith(f) for f in self.fail_on):
            raise RuntimeError("boom")
        return TranscriptionResult(segments=[Segment(0.0, 1.0, "hi")], language="en")


def _run_batch(monkeypatch, tmp_path, names, model_factory):
    service = ts.TranscriptionService()
    model = model_factory(service)
    monkeypatch.setattr(service, "_load_model", lambda *a, **k: model)
    monkeypatch.setattr(ts.chunking, "probe_duration_seconds", lambda p: 5.0)
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x")
        paths.append(str(p))
    finished, outcomes, errors = [], {}, []
    service.transcribe_batch(
        paths, "tiny", {"srt"}, lambda m: None, lambda t: None,
        on_error=lambda m, tb: errors.append(m),
        on_item_complete=lambda p, o: outcomes.__setitem__(p, o),
        on_finished=finished.append,
    )
    return paths, finished, outcomes, errors


def test_cancel_during_last_item_still_finishes(monkeypatch, tmp_path):
    # Regression: cancelling the only/last file used to leave the GUI stuck
    # (no terminal status was ever emitted).
    paths, finished, outcomes, _ = _run_batch(
        monkeypatch, tmp_path, ["only.wav"], lambda s: _FakeModel(service=s, cancel_on_call=1))
    assert len(finished) == 1 and finished[0].startswith("Cancelled")
    assert outcomes[paths[0]] == ts.OUTCOME_CANCELLED


def test_failed_item_is_not_reported_done(monkeypatch, tmp_path):
    paths, finished, outcomes, errors = _run_batch(
        monkeypatch, tmp_path, ["good.wav", "bad.wav"], lambda s: _FakeModel(fail_on={"bad.wav"}))
    assert outcomes[paths[0]] == ts.OUTCOME_DONE
    assert outcomes[paths[1]] == ts.OUTCOME_ERROR
    assert len(errors) == 1
    assert "1 failed" in finished[0]
    assert (tmp_path / "good.srt").exists()


def test_batch_with_clashing_stems_writes_both(monkeypatch, tmp_path):
    _run_batch(monkeypatch, tmp_path, ["talk.mp3", "talk.mp4"], lambda s: _FakeModel())
    assert (tmp_path / "talk.mp3.srt").exists()
    assert (tmp_path / "talk.mp4.srt").exists()


def test_on_finished_called_when_model_fails_to_load(monkeypatch, tmp_path):
    service = ts.TranscriptionService()
    monkeypatch.setattr(service, "_load_model", lambda *a, **k: None)
    finished = []
    service.transcribe_batch([str(tmp_path / "a.wav")], "tiny", {"srt"}, lambda m: None, lambda t: None,
                             on_finished=finished.append)
    assert len(finished) == 1 and finished[0].startswith("Error")


def test_url_title_with_dot_keeps_full_name(monkeypatch, tmp_path):
    service = ts.TranscriptionService()
    monkeypatch.setattr(ts.chunking, "probe_duration_seconds", lambda p: 5.0)
    monkeypatch.setattr(ts.remote_input, "download_audio", lambda url, d: (str(tmp_path / "a.wav"), "Episode 1.5"))
    monkeypatch.setattr(ts.Path, "home", lambda: tmp_path)
    outcome = service._process_file(_FakeModel(), "https://example.com/v", "tiny", {"srt"},
                                    lambda m: None, lambda t: None, None, 15)
    assert outcome == ts.OUTCOME_DONE
    assert (tmp_path / "Downloads" / "Episode 1.5.srt").exists()
