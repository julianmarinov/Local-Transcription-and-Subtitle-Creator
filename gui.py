"""Tkinter GUI. Business logic lives in transcription_service.py."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

import chunking
import remote_input
from backends import MODEL_SIZES, cached_backend
from transcription_service import (
    AUDIO_VIDEO_EXTENSIONS,
    OUTCOME_CANCELLED,
    OUTCOME_DONE,
    TranscriptionService,
)

_AUDIO_VIDEO_SUFFIXES = {ext.replace("*", "") for ext in AUDIO_VIDEO_EXTENSIONS}

# Shown in the Language box. "Auto-detect" lets Whisper guess from the
# first 30 seconds, which can go wrong on clips that open with music or
# silence - picking the language explicitly avoids that and is slightly
# faster.
_LANGUAGES = {
    "Auto-detect": None,
    "Bulgarian": "bg",
    "English": "en",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
    "Italian": "it",
    "Russian": "ru",
    "Greek": "el",
    "Turkish": "tr",
    "Romanian": "ro",
    "Serbian": "sr",
    "Macedonian": "mk",
    "Ukrainian": "uk",
    "Polish": "pl",
    "Portuguese": "pt",
    "Dutch": "nl",
}

_DEFAULT_MODEL = "turbo"


class AudioToSRTConverter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local Transcription & Subtitle Creator")
        self.geometry("820x680")
        self.transcriber = TranscriptionService()
        self._queue_rows: dict[str, str] = {}  # input_path -> Treeview item id
        self._running = False
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Model selector
        ttk.Label(frm, text="Model:").grid(row=0, column=0, sticky="e")
        self.model_var = tk.StringVar(value=_DEFAULT_MODEL)
        model_combo = ttk.Combobox(frm, textvariable=self.model_var,
                                    values=MODEL_SIZES, state="readonly", width=12)
        model_combo.grid(row=0, column=1, sticky="w", padx=5)

        # --- Output formats
        ttk.Label(frm, text="Formats:").grid(row=0, column=2, sticky="e")
        fmt_frm = ttk.Frame(frm)
        fmt_frm.grid(row=0, column=3, sticky="w")
        self.format_vars = {
            "srt": tk.BooleanVar(value=True),
            "vtt": tk.BooleanVar(value=False),
            "txt": tk.BooleanVar(value=False),
            "json": tk.BooleanVar(value=False),
        }
        for fmt, var in self.format_vars.items():
            ttk.Checkbutton(fmt_frm, text=f".{fmt}", variable=var).pack(side="left", padx=(0, 8))

        # --- Chunk length (for auto-splitting long files)
        ttk.Label(frm, text="Chunk size (min):").grid(row=1, column=0, sticky="e", pady=(5, 0))
        self.chunk_minutes_var = tk.IntVar(value=int(chunking.DEFAULT_CHUNK_MINUTES))
        ttk.Spinbox(frm, from_=5, to=60, textvariable=self.chunk_minutes_var, width=5).grid(
            row=1, column=1, sticky="w", padx=5, pady=(5, 0))

        # --- Spoken language
        ttk.Label(frm, text="Language:").grid(row=1, column=2, sticky="e", pady=(5, 0))
        self.language_var = tk.StringVar(value="Auto-detect")
        ttk.Combobox(frm, textvariable=self.language_var, values=list(_LANGUAGES),
                     state="readonly", width=14).grid(row=1, column=3, sticky="w", pady=(5, 0))

        # --- Queue
        queue_frm = ttk.Frame(frm)
        queue_frm.grid(row=2, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        frm.rowconfigure(2, weight=1)
        for c in range(4):
            frm.columnconfigure(c, weight=1)

        self.queue_tree = ttk.Treeview(queue_frm, columns=("status",), show="tree headings", height=8)
        self.queue_tree.heading("#0", text="File")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.column("#0", width=420)
        self.queue_tree.column("status", width=260)
        scrollbar = ttk.Scrollbar(queue_frm, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=scrollbar.set)
        self.queue_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        queue_btn_frm = ttk.Frame(frm)
        queue_btn_frm.grid(row=3, column=0, columnspan=4, sticky="ew", pady=5)
        ttk.Button(queue_btn_frm, text="Add Files...", command=self._add_files).pack(side="left", padx=(0, 5))
        ttk.Button(queue_btn_frm, text="Add Folder...", command=self._add_folder).pack(side="left", padx=(0, 5))
        ttk.Button(queue_btn_frm, text="Add YouTube URL...", command=self._add_url).pack(side="left", padx=(0, 5))
        ttk.Button(queue_btn_frm, text="Remove Selected", command=self._remove_selected).pack(side="left", padx=(0, 5))
        ttk.Button(queue_btn_frm, text="Clear Queue", command=self._clear_queue).pack(side="left")

        # --- Buttons
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=4, column=0, columnspan=4, pady=10)
        self.go_btn = ttk.Button(btn_frm, text="Transcribe Queue", command=self._on_transcribe)
        self.go_btn.pack(side="left", padx=5)
        self.cancel_btn = ttk.Button(btn_frm, text="Cancel", command=self._on_cancel, state="disabled")
        self.cancel_btn.pack(side="left")

        # --- Status + spinner
        self.status_lbl = ttk.Label(frm, text="Ready")
        self.status_lbl.grid(row=5, column=0, columnspan=4, sticky="w")
        self.spinner = ttk.Progressbar(frm, mode="indeterminate")
        self.spinner.grid(row=6, column=0, columnspan=4, sticky="ew", pady=5)

        ttk.Button(frm, text="Help", command=self._open_help_window).grid(
            row=7, column=0, columnspan=4, pady=(5, 0), sticky="ew")

        # --- Transcribed text (last completed file)
        self.txt = scrolledtext.ScrolledText(self, height=12, state="disabled")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    def _open_help_window(self) -> None:
        help_window = tk.Toplevel(self)
        help_window.title("Help Information")
        help_window.geometry("650x280")

        try:
            backend = cached_backend()
            device = backend.device_label()
        except Exception as e:
            # Show the failure instead of leaving the window blank -
            # backend selection can fail (e.g. a broken optional install).
            ttk.Label(
                help_window,
                text=f"Could not determine the active backend:\n\n{e}",
                justify=tk.LEFT, foreground="red",
            ).pack(padx=10, pady=10)
            return

        help_text = f"""
What this script does?
Add one or more audio/video files (or a whole folder, or a YouTube URL) to the queue,
pick a model and output format(s), and click "Transcribe Queue". Each file's subtitles
are written next to the source file; a YouTube URL's audio is downloaded locally first
(one-time network fetch) and its subtitles are saved to ~/Downloads, named after the
video's title - transcription itself always runs entirely on-device.

Which model to choose?
"turbo" (default) is nearly as accurate as "large" and several times faster - the best choice
for most videos. "large" is the most accurate but slowest. "tiny"/"base" are quick drafts only.
Set Language if you know it - auto-detection can guess wrong on clips that start with music.

Selected backend: {backend.name}
Detected compute device: {device}
{"(NVIDIA GPU acceleration)" if "cuda" in device else "(Apple Silicon GPU acceleration)" if "mps" in device else "(CPU only - this will be significantly slower, especially for larger models)"}

Long files (over 20 minutes) are automatically split into chunks (size configurable above)
so memory usage stays bounded and progress updates incrementally.

- Supported input formats: MP3, WAV, M4A, AAC, FLAC, OGG, OPUS, WMA (audio),
  MP4, MOV, MKV, WEBM, AVI, M4V (video - audio track is extracted automatically)
- Existing subtitle files with the same name are overwritten
"""
        ttk.Label(help_window, text=help_text, justify=tk.LEFT).pack(padx=10, pady=10)

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select Audio or Video Files",
            filetypes=[("Audio/Video", " ".join(AUDIO_VIDEO_EXTENSIONS)), ("All files", "*.*")]
        )
        for p in paths:
            self._add_to_queue(p)

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Folder")
        if not folder:
            return
        added = 0
        for path in sorted(Path(folder).rglob("*")):
            if path.name.startswith("."):  # macOS "._name" AppleDouble junk, hidden files
                continue
            if path.is_file() and path.suffix.lower() in _AUDIO_VIDEO_SUFFIXES:
                added += self._add_to_queue(str(path))
        if not added:
            messagebox.showinfo("Nothing added", "No new audio or video files were found in that folder.")

    def _add_url(self) -> None:
        url = simpledialog.askstring("Add YouTube URL", "Enter a video URL:", parent=self)
        if not url:
            return
        url = url.strip()
        if not remote_input.is_url(url):
            messagebox.showerror("Invalid URL", "That doesn't look like a valid http(s) URL.")
            return
        self._add_to_queue(url)

    def _add_to_queue(self, path: str) -> bool:
        if path in self._queue_rows:
            return False
        display = path if remote_input.is_url(path) else os.path.basename(path)
        iid = self.queue_tree.insert("", "end", text=display, values=("Queued",))
        self._queue_rows[path] = iid
        return True

    def _remove_selected(self) -> None:
        if self._running:
            messagebox.showinfo("Busy", "Wait for the current run to finish (or cancel it) before changing the queue.")
            return
        for iid in self.queue_tree.selection():
            path = next((p for p, row_iid in self._queue_rows.items() if row_iid == iid), None)
            if path:
                del self._queue_rows[path]
            self.queue_tree.delete(iid)

    def _clear_queue(self) -> None:
        if self._running:
            messagebox.showinfo("Busy", "Wait for the current run to finish (or cancel it) before changing the queue.")
            return
        self.queue_tree.delete(*self.queue_tree.get_children())
        self._queue_rows.clear()

    def _on_transcribe(self) -> None:
        input_paths = list(self._queue_rows.keys())
        if not input_paths:
            messagebox.showwarning("Empty queue", "Add at least one audio or video file to the queue.")
            return

        formats = {fmt for fmt, var in self.format_vars.items() if var.get()}
        if not formats:
            messagebox.showwarning("No format selected", "Please select at least one output format.")
            return

        model = self.model_var.get()
        language = _LANGUAGES.get(self.language_var.get())
        try:
            chunk_minutes = max(1, min(120, int(self.chunk_minutes_var.get())))
        except (tk.TclError, ValueError):
            chunk_minutes = int(chunking.DEFAULT_CHUNK_MINUTES)
            self.chunk_minutes_var.set(chunk_minutes)

        for path in input_paths:
            self._set_row_status(path, "Queued")

        self._set_running(True)

        status_cb = lambda msg: self.after(0, self._update_status, msg)
        text_cb = lambda txt: self.after(0, self._update_text, txt)
        error_cb = lambda msg, tb: self.after(0, self._show_error, msg, tb)
        item_status_cb = lambda path, msg: self.after(0, self._set_row_status, path, msg)
        item_complete_cb = lambda path, outcome: self.after(0, self._on_item_complete, path, outcome)
        finished_cb = lambda summary: self.after(0, self._on_finished, summary)

        t = threading.Thread(
            target=self.transcriber.transcribe_batch,
            args=(input_paths, model, formats, status_cb, text_cb, error_cb, item_status_cb, item_complete_cb,
                  chunk_minutes, language, finished_cb),
            daemon=True
        )
        t.start()

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.go_btn.config(state="disabled" if running else "normal")
        self.cancel_btn.config(state="normal" if running else "disabled")
        if running:
            self.spinner.start(10)
        else:
            self.spinner.stop()

    def _on_item_complete(self, path: str, outcome: str) -> None:
        # Only overwrite the row on success/cancel - an error row keeps
        # its "Error: ..." text so the user can see which file failed.
        if outcome == OUTCOME_DONE:
            iid = self._queue_rows.get(path)
            current = self.queue_tree.set(iid, "status") if iid and self.queue_tree.exists(iid) else ""
            if not current.startswith("Done"):
                self._set_row_status(path, "Done")
        elif outcome == OUTCOME_CANCELLED:
            self._set_row_status(path, "Cancelled")

    def _on_finished(self, summary: str) -> None:
        self._set_running(False)
        self.status_lbl.config(text=summary)

    def _set_row_status(self, path: str, status: str) -> None:
        iid = self._queue_rows.get(path)
        if iid and self.queue_tree.exists(iid):
            self.queue_tree.set(iid, "status", status)

    def _on_cancel(self) -> None:
        if not self._running:
            return
        self.transcriber.cancel_transcription()
        self.cancel_btn.config(state="disabled")
        self._update_status("Cancelling - stops after the current part finishes...")

    def _on_close(self) -> None:
        if self._running and not messagebox.askokcancel(
            "Transcription running", "A transcription is still running. Quit anyway?"
        ):
            return
        self.transcriber.cancel_transcription()
        self.destroy()

    def _update_status(self, msg: str) -> None:
        self.status_lbl.config(text=msg)

    def _update_text(self, txt: str) -> None:
        self.txt.config(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, txt)
        self.txt.config(state="disabled")

    def _show_error(self, msg: str, tb: str) -> None:
        # The full traceback is noise for most users but essential for a
        # bug report - keep the dialog short and put the tail in "detail".
        tail = "\n".join(tb.strip().splitlines()[-12:])
        messagebox.showerror("Transcription Error", msg, detail=tail)
