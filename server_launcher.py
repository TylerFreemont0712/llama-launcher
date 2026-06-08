#!/usr/bin/env python3
"""
server_launcher.py
Unified PyQt6 launcher for local AI services.

Tabs:
  - llama.cpp   (HTTP server for GGUF text models)
  - Whisper     (FastAPI transcription server, see whisper_server.py)

Each tab manages its own subprocess via QProcess and shares the
same dark theme.
"""

from __future__ import annotations

import os
import shutil
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QColor, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QSpinBox, QStatusBar, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

HOME       = Path.home()
MODELS_DIR = HOME / "ai" / "models"
LLAMA_DIR  = HOME / "llama.cpp"

LLAMA_SERVER_CANDIDATES = [
    LLAMA_DIR / "build" / "bin" / "llama-server",
    LLAMA_DIR / "build" / "bin" / "server",
    LLAMA_DIR / "llama-server",
    LLAMA_DIR / "server",
]

# whisper_server.py is expected next to this file by default.
DEFAULT_WHISPER_SCRIPT = Path(__file__).resolve().parent / "whisper_server.py"
DEFAULT_WHISPER_OUTPUT = "~/Program/Whisper-Audio/Results"

# ─── Theme ────────────────────────────────────────────────────────────────────

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT_MAIN = "#eaeaea"
TEXT_DIM  = "#888"
BORDER    = "#2a2a4a"
LOG_BG    = "#0d0d1a"
BTN_START = "#1a6b3c"
BTN_STOP  = "#7a1a1a"
BTN_HOV_S = "#22913e"
BTN_HOV_X = "#a01a1a"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_MAIN};
    font-family: 'JetBrains Mono', 'Fira Mono', 'Consolas', monospace;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: {HIGHLIGHT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px; top: -1px;
    padding: 0 4px;
}}
QLabel {{ color: {TEXT_MAIN}; }}
QComboBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_MAIN};
    min-height: 26px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    color: {TEXT_MAIN};
}}
QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_MAIN};
    min-height: 26px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {ACCENT}; border: none; width: 16px;
}}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {PANEL_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {HIGHLIGHT}; border-color: {HIGHLIGHT};
}}
QPushButton {{
    border-radius: 5px; padding: 7px 18px;
    font-weight: bold; font-size: 13px; min-height: 32px;
}}
QPushButton#startBtn {{
    background-color: {BTN_START}; color: #fff; border: none;
}}
QPushButton#startBtn:hover {{ background-color: {BTN_HOV_S}; }}
QPushButton#startBtn:disabled {{ background-color: #2a2a2a; color: #555; }}
QPushButton#stopBtn {{
    background-color: {BTN_STOP}; color: #fff; border: none;
}}
QPushButton#stopBtn:hover {{ background-color: {BTN_HOV_X}; }}
QPushButton#stopBtn:disabled {{ background-color: #2a2a2a; color: #555; }}
QPushButton#refreshBtn {{
    background-color: {ACCENT}; color: {TEXT_MAIN}; border: none;
    padding: 4px 10px; font-size: 12px; min-height: 26px; border-radius: 4px;
}}
QPushButton#refreshBtn:hover {{ background-color: #1a4a80; }}
QPushButton#clearBtn, QPushButton#copyBtn {{
    background-color: #2a2a3a; color: {TEXT_DIM}; border: none;
    padding: 4px 10px; font-size: 12px; min-height: 26px;
}}
QPushButton#clearBtn:hover, QPushButton#copyBtn:hover {{
    color: {TEXT_MAIN}; background-color: #3a3a4a;
}}
QPushButton#browserBtn {{
    background-color: #1a3a5a; color: {TEXT_MAIN}; border: none;
    padding: 7px 14px; font-size: 13px; min-height: 32px;
    border-radius: 5px; font-weight: bold;
}}
QPushButton#browserBtn:hover {{ background-color: #1a4a7a; }}
QPushButton#browserBtn:disabled {{ background-color: #2a2a2a; color: #555; }}
QTextEdit {{
    background-color: {LOG_BG}; color: #aaffaa;
    border: 1px solid {BORDER}; border-radius: 4px;
    font-family: 'JetBrains Mono', 'Fira Mono', 'Consolas', monospace;
    font-size: 12px; padding: 4px;
}}
QStatusBar {{
    background-color: {PANEL_BG}; color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}
QScrollBar:vertical {{
    background: {PANEL_BG}; width: 8px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {ACCENT}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    top: -1px;
    background: {DARK_BG};
}}
QTabBar::tab {{
    background: {PANEL_BG};
    color: {TEXT_DIM};
    padding: 8px 20px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
    font-weight: bold;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: {HIGHLIGHT};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_MAIN};
    background: #1a2a4a;
}}
"""

# ─── llama.cpp parameter definitions ──────────────────────────────────────────

LLAMA_PARAMS = [
    # (label, flag, default, min, max, step, tooltip)
    ("Context Size",   "--ctx-size",    4096, 512,  131072, 512,
     "Max tokens in the KV cache / context window.\n"
     "Higher = more memory. Common: 2048, 4096, 8192."),
    ("GPU Layers",     "-ngl",          0,    0,    999,    1,
     "Number of model layers to offload to GPU.\n"
     "Set to 999 to offload everything. 0 = CPU only."),
    ("Batch Size",     "--batch-size",  512,  32,   4096,   64,
     "Prompt processing batch size.\n"
     "Larger = faster prompt ingestion, more memory."),
    ("UBatch Size",    "--ubatch-size", 512,  32,   4096,   64,
     "Physical micro-batch size.\nUsually same as batch-size."),
    ("Threads",        "--threads",     max(1, (os.cpu_count() or 4) // 2),
                                        1,    256,  1,
     "CPU threads for inference.\n"
     "Rule of thumb: number of physical (not logical) cores."),
    ("Parallel Slots", "--parallel",    1,    1,    64,     1,
     "Simultaneous request slots.\n"
     "Increase only for concurrent API clients."),
    ("Port",           "--port",        8080, 1024, 65535,  1,
     "HTTP port the server listens on.\nDefault: 8080."),
]

LLAMA_CHECKBOXES = [
    ("Flash Attention", "--flash-attn",
     "Enable FlashAttention — faster inference, requires compatible GPU/build."),
    ("mlock",           "--mlock",
     "Lock model weights in RAM, prevents swapping to disk."),
    ("No mmap",         "--no-mmap",
     "Load model into RAM instead of memory-mapping.\n"
     "Slower startup, sometimes faster inference."),
    ("Verbose",         "--verbose",
     "Print extra debug info into the log."),
    ("Embeddings",      "--embeddings",
     "Enable the /embedding endpoint for vector generation."),
]

LLAMA_STRING_PARAMS = [
    ("--host", "Host",       "127.0.0.1",
     "IP to bind to.\n127.0.0.1 = local only.  0.0.0.0 = all interfaces."),
    (None,     "Extra Args", "",
     "Extra raw arguments to append.\nExample: --rope-scale 2.0 --log-disable"),
]

# ─── Whisper parameter definitions ────────────────────────────────────────────

WHISPER_INT_PARAMS = [
    # (label, flag, default, min, max, step, tooltip)
    ("Port",      "--port",      8000, 1024, 65535, 1,
     "HTTP port (default 8000 — kept distinct from llama.cpp's 8080)."),
    ("Beam Size", "--beam-size", 10,   1,    20,    1,
     "Beam search width. Higher = more accurate, slower.\n"
     "Tuned high for accuracy."),
    ("Best Of",   "--best-of",   10,   1,    20,    1,
     "Number of candidates considered. Higher = more accurate."),
]

WHISPER_FLOAT_PARAMS = [
    ("Patience", "--patience", 2.0, 0.5, 5.0, 0.5, 2,
     "Beam search patience. Higher lets beam search explore more.\n"
     "1.0 = standard, 2.0 = generous (default here)."),
]

WHISPER_CHECKBOXES = [
    ("VAD Filter",        "--vad-filter",
     "Filter non-speech segments via Silero VAD before transcribing."),
    ("Condition on Prev", "--condition-on-previous",
     "Feed prior segment text as context — improves consistency."),
    ("Word Timestamps",   "--word-timestamps",
     "Generate per-word timestamps (slower)."),
]

# Inverse flags for explicit "off" — we always pass one of --foo / --no-foo
WHISPER_CHECKBOX_NEGATIVE = {
    "--vad-filter":            "--no-vad-filter",
    "--condition-on-previous": "--no-condition-on-previous",
    "--word-timestamps":       "--no-word-timestamps",
}

WHISPER_CHECKBOX_DEFAULTS = {
    "--vad-filter":            True,
    "--condition-on-previous": True,
    "--word-timestamps":       False,
}

WHISPER_COMBOS = [
    # (label, flag, choices, default, tooltip)
    ("Model",        "--model",
     ["large-v3", "large-v2", "medium", "small", "base"], "large-v3",
     "Whisper model. large-v3 is the most accurate."),
    ("Device",       "--device",
     ["cuda", "cpu", "auto"], "cuda",
     "Compute device for inference."),
    ("Compute Type", "--compute-type",
     ["float16", "int8_float16", "int8", "float32"], "float16",
     "Numerical precision.\n"
     "float16 = max accuracy on a 8GB+ GPU with this model.\n"
     "int8_float16 = lower VRAM, slightly less accurate."),
    ("Language",     "--language",
     ["ja", "en", "zh", "ko", "auto"], "ja",
     "Source language. ja for Japanese."),
]

WHISPER_STRING_PARAMS = [
    # (flag, label, default, tooltip)
    ("--host",       "Host",
     "127.0.0.1", "Bind address. 127.0.0.1 = localhost only."),
    ("--output-dir", "Output Dir",
     DEFAULT_WHISPER_OUTPUT,
     "Directory where transcripts are saved as .txt files."),
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_llama_binary() -> str:
    for p in LLAMA_SERVER_CANDIDATES:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which("llama-server") or shutil.which("server") or ""


def scan_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    found: list[Path] = []
    for pat in ("**/*.gguf", "**/*.GGUF"):
        found.extend(MODELS_DIR.glob(pat))
    return sorted(set(found))


# ─── Base widget: shared process + log machinery ──────────────────────────────

class BaseServerWidget(QWidget):
    """Shared process management and log handling for server tabs."""

    def __init__(self, tab_label: str):
        super().__init__()
        self.tab_label = tab_label
        self.process: Optional[QProcess] = None

        # Subclasses must populate these before calling _finalize_actions():
        self.cmd_preview: Optional[QLabel] = None
        self.start_btn: Optional[QPushButton] = None
        self.stop_btn: Optional[QPushButton] = None
        self.browser_btn: Optional[QPushButton] = None
        self.log_edit: Optional[QTextEdit] = None
        self.log_status: Optional[QLabel] = None

    # --- subclasses override these ------------------------------------------
    def _build_command(self) -> list[str]:
        raise NotImplementedError

    def _open_browser_url(self) -> str:
        raise NotImplementedError

    def _validate_before_start(self) -> Optional[str]:
        """Return None if OK, or an error message if start should abort."""
        return None

    # --- shared helpers -----------------------------------------------------
    def _log_section(self) -> QGroupBox:
        grp = QGroupBox("Server Log")
        vl  = QVBoxLayout(grp)
        vl.setSpacing(4)

        bar    = QWidget()
        bar_hl = QHBoxLayout(bar)
        bar_hl.setContentsMargins(0, 0, 0, 0)
        self.log_status = QLabel("● Stopped")
        self.log_status.setStyleSheet(f"color: {TEXT_DIM};")
        bar_hl.addWidget(self.log_status)
        bar_hl.addStretch()
        clr = QPushButton("Clear Log")
        clr.setObjectName("clearBtn")
        clr.clicked.connect(self._clear_log)
        bar_hl.addWidget(clr)
        vl.addWidget(bar)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("JetBrains Mono", 11))
        vl.addWidget(self.log_edit)
        return grp

    def _action_bar(self) -> QWidget:
        w  = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        self.cmd_preview = QLabel("")
        self.cmd_preview.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.cmd_preview.setWordWrap(True)
        self.cmd_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        hl.addWidget(self.cmd_preview, stretch=1)

        copy_btn = QPushButton("⎘ Copy")
        copy_btn.setObjectName("copyBtn")
        copy_btn.setToolTip("Copy full server command to clipboard")
        copy_btn.clicked.connect(self._copy_command)
        hl.addWidget(copy_btn)

        self.browser_btn = QPushButton("⌁ Open UI")
        self.browser_btn.setObjectName("browserBtn")
        self.browser_btn.clicked.connect(self._open_browser)
        self.browser_btn.setEnabled(False)
        hl.addWidget(self.browser_btn)

        self.start_btn = QPushButton("▶  Start Server")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumWidth(160)
        self.start_btn.clicked.connect(self._start_server)
        hl.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop Server")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setMinimumWidth(140)
        self.stop_btn.clicked.connect(self._stop_server)
        hl.addWidget(self.stop_btn)

        return w

    def _start_server(self):
        err = self._validate_before_start()
        if err:
            QMessageBox.warning(self, "Cannot Start", err)
            return
        cmd = self._build_command()
        if not cmd:
            QMessageBox.warning(self, "Cannot Start", "Empty command.")
            return

        self._log(f"$ {' '.join(cmd)}\n", color="#e9c46a")
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyRead.connect(self._on_output)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.start(cmd[0], cmd[1:])

        if not self.process.waitForStarted(3000):
            self._log("ERROR: Process failed to start.\n", color=HIGHLIGHT)
            self.process = None
        else:
            pid = self.process.processId()
            self.log_status.setText(f"● Running  (PID {pid})")
            self.log_status.setStyleSheet("color: #44ff88; font-weight: bold;")
        self._update_buttons()

    def _stop_server(self):
        if self.process:
            self._log("\n[Stopping server…]\n", color="#e9c46a")
            self.process.terminate()
            if not self.process.waitForFinished(4000):
                self.process.kill()
                self.process.waitForFinished(1000)
            self.process = None
        self._update_buttons()

    def _on_output(self):
        if self.process:
            data = bytes(self.process.readAll()).decode("utf-8", errors="replace")
            self._log(data)

    def _on_process_finished(self, exit_code, _exit_status):
        self._log(f"\n[Process exited — code {exit_code}]\n", color="#e9c46a")
        self.process = None
        self.log_status.setText("● Stopped")
        self.log_status.setStyleSheet(f"color: {TEXT_DIM};")
        self._update_buttons()

    def _on_process_error(self, error):
        labels = {
            QProcess.ProcessError.FailedToStart: "FailedToStart",
            QProcess.ProcessError.Crashed:       "Crashed",
            QProcess.ProcessError.Timedout:      "Timedout",
        }
        self._log(f"\n[Process error: {labels.get(error, 'UnknownError')}]\n",
                  color=HIGHLIGHT)
        self.process = None
        self._update_buttons()

    def _update_buttons(self):
        running = self.is_running()
        if self.start_btn:   self.start_btn.setEnabled(not running)
        if self.stop_btn:    self.stop_btn.setEnabled(running)
        if self.browser_btn: self.browser_btn.setEnabled(running)

    def is_running(self) -> bool:
        return (self.process is not None
                and self.process.state() != QProcess.ProcessState.NotRunning)

    def _copy_command(self):
        cmd = self._build_command()
        if cmd:
            QApplication.clipboard().setText(" ".join(cmd))

    def _open_browser(self):
        try:
            webbrowser.open(self._open_browser_url())
        except Exception as e:
            self._log(f"\n[Failed to open browser: {e}]\n", color=HIGHLIGHT)

    def _update_preview(self):
        cmd = self._build_command()
        if self.cmd_preview is not None:
            self.cmd_preview.setText(
                f"<span style='color:{TEXT_DIM}'>$ {' '.join(cmd)}</span>"
                if cmd else ""
            )

    def _log(self, text: str, color: Optional[str] = None):
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if color:
            fmt = cursor.charFormat()
            fmt.setForeground(QColor(color))
            cursor.setCharFormat(fmt)
            cursor.insertText(text)
            fmt.setForeground(QColor("#aaffaa"))
            cursor.setCharFormat(fmt)
        else:
            cursor.insertText(text)
        self.log_edit.setTextCursor(cursor)
        self.log_edit.ensureCursorVisible()

    def _clear_log(self):
        if self.log_edit:
            self.log_edit.clear()


# ─── llama.cpp tab ────────────────────────────────────────────────────────────

class LlamaServerWidget(BaseServerWidget):
    def __init__(self):
        super().__init__("llama.cpp")
        self.param_widgets:    dict[str, QSpinBox]  = {}
        self.checkbox_widgets: dict[str, QCheckBox] = {}
        self.str_widgets:      dict[Optional[str], QLineEdit] = {}

        self._build_ui()
        self._refresh_models()
        self._wire_preview_signals()
        self._update_preview()
        self._update_buttons()

    # --- UI -----------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._header())
        root.addWidget(self._model_section())
        root.addWidget(self._params_section())
        root.addWidget(self._server_section())
        root.addWidget(self._action_bar())
        root.addWidget(self._log_section(), stretch=1)

    def _header(self) -> QWidget:
        w  = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("🦙  llama.cpp Server")
        title.setFont(QFont("JetBrains Mono", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {HIGHLIGHT};")
        hl.addWidget(title)
        hl.addStretch()
        info = QLabel(f"models: {MODELS_DIR}  |  bin: {LLAMA_DIR}")
        info.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        hl.addWidget(info)
        return w

    def _model_section(self) -> QGroupBox:
        grp = QGroupBox("Model Selection")
        hl  = QHBoxLayout(grp)
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.model_combo.setToolTip("GGUF model files found under ~/ai/models/")
        hl.addWidget(self.model_combo)
        ref = QPushButton("⟳ Refresh")
        ref.setObjectName("refreshBtn")
        ref.clicked.connect(self._refresh_models)
        hl.addWidget(ref)
        return grp

    def _params_section(self) -> QGroupBox:
        grp  = QGroupBox("Server Parameters")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        col_count = 3

        for i, (label, flag, default, mn, mx, step, tip) in enumerate(LLAMA_PARAMS):
            row, base_col = i // col_count, (i % col_count) * 3
            lbl = QLabel(label + ":")
            lbl.setToolTip(tip)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            spin = QSpinBox()
            spin.setRange(mn, mx); spin.setValue(default); spin.setSingleStep(step)
            spin.setToolTip(tip); spin.setMinimumWidth(90)
            grid.addWidget(lbl,  row, base_col)
            grid.addWidget(spin, row, base_col + 1)
            if base_col + 2 < col_count * 3 - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet(f"color: {BORDER};")
                grid.addWidget(sep, row, base_col + 2)
            self.param_widgets[flag] = spin

        cb_row = (len(LLAMA_PARAMS) + col_count - 1) // col_count
        cb_grp = QGroupBox("Flags")
        cb_grp.setStyleSheet(
            f"QGroupBox {{ color: {TEXT_DIM}; border: 1px solid {BORDER}; }}"
        )
        cb_hl  = QHBoxLayout(cb_grp)
        cb_hl.setSpacing(18)
        for label, flag, tip in LLAMA_CHECKBOXES:
            cb = QCheckBox(label)
            cb.setToolTip(tip)
            cb_hl.addWidget(cb)
            self.checkbox_widgets[flag] = cb
        cb_hl.addStretch()
        grid.addWidget(cb_grp, cb_row, 0, 1, col_count * 3)
        return grp

    def _server_section(self) -> QGroupBox:
        grp  = QGroupBox("Connection & Extra")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        for row, (flag, label, default, tip) in enumerate(LLAMA_STRING_PARAMS):
            lbl = QLabel(label + ":")
            lbl.setToolTip(tip)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            le  = QLineEdit(default)
            le.setToolTip(tip)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(le,  row, 1, 1, 3)
            self.str_widgets[flag] = le

        lbl_bin = QLabel("Binary Path:")
        lbl_bin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.binary_path_edit = QLineEdit(find_llama_binary())
        grid.addWidget(lbl_bin,               len(LLAMA_STRING_PARAMS), 0)
        grid.addWidget(self.binary_path_edit, len(LLAMA_STRING_PARAMS), 1, 1, 3)
        grid.setColumnStretch(1, 1)
        return grp

    # --- model scan ---------------------------------------------------------
    def _refresh_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        models = scan_models()
        if not models:
            self.model_combo.addItem("(no .gguf files found in ~/ai/models)")
        else:
            for p in models:
                try:
                    display = str(p.relative_to(MODELS_DIR))
                except ValueError:
                    display = str(p)
                try:
                    size_str = f"{p.stat().st_size / (1024**3):.2f} GB"
                except OSError:
                    size_str = "? GB"
                self.model_combo.addItem(f"{display}  [{size_str}]", userData=str(p))
        self.model_combo.blockSignals(False)
        self._update_preview()

    # --- command building ---------------------------------------------------
    def _build_command(self) -> list[str]:
        binary = self.binary_path_edit.text().strip()
        if not binary:
            return []
        cmd = [binary]
        model_path = self.model_combo.currentData()
        if model_path:
            cmd += ["--model", model_path]
        host = self.str_widgets.get("--host")
        if host and host.text().strip():
            cmd += ["--host", host.text().strip()]
        for flag, widget in self.param_widgets.items():
            cmd += [flag, str(widget.value())]
        for flag, widget in self.checkbox_widgets.items():
            if widget.isChecked():
                cmd.append(flag)
        extra = self.str_widgets.get(None)
        if extra and extra.text().strip():
            cmd += extra.text().strip().split()
        return cmd

    def _validate_before_start(self) -> Optional[str]:
        binary = self.binary_path_edit.text().strip()
        if not binary:
            return "Server binary path is empty.\nSet it in Connection & Extra."
        if not self.model_combo.currentData():
            return "No model selected."
        if not os.path.isfile(binary):
            return f"Cannot find binary:\n{binary}\n\nBuild llama.cpp first."
        return None

    def _open_browser_url(self) -> str:
        host = self.str_widgets.get("--host")
        port = self.param_widgets.get("--port")
        h = (host.text().strip() if host else "") or "127.0.0.1"
        if h == "0.0.0.0":
            h = "127.0.0.1"
        p = port.value() if port else 8080
        return f"http://{h}:{p}"

    def _wire_preview_signals(self):
        for w in self.param_widgets.values():
            w.valueChanged.connect(self._update_preview)
        for w in self.checkbox_widgets.values():
            w.stateChanged.connect(self._update_preview)
        for w in self.str_widgets.values():
            w.textChanged.connect(self._update_preview)
        self.binary_path_edit.textChanged.connect(self._update_preview)
        self.model_combo.currentIndexChanged.connect(self._update_preview)


# ─── Whisper tab ──────────────────────────────────────────────────────────────

@dataclass
class WhisperWidgets:
    int_spins:     dict
    float_spins:   dict
    checkboxes:    dict
    combos:        dict
    string_inputs: dict


class WhisperServerWidget(BaseServerWidget):
    def __init__(self):
        super().__init__("whisper")
        self.w = WhisperWidgets({}, {}, {}, {}, {})

        self._build_ui()
        self._wire_preview_signals()
        self._update_preview()
        self._update_buttons()

    # --- UI -----------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._header())
        root.addWidget(self._model_section())
        root.addWidget(self._transcription_section())
        root.addWidget(self._server_section())
        root.addWidget(self._action_bar())
        root.addWidget(self._log_section(), stretch=1)

    def _header(self) -> QWidget:
        w  = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("🎙️  Whisper Transcription Server")
        title.setFont(QFont("JetBrains Mono", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {HIGHLIGHT};")
        hl.addWidget(title)
        hl.addStretch()
        info = QLabel("POST /transcribe  →  saves .txt to output dir")
        info.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        hl.addWidget(info)
        return w

    def _model_section(self) -> QGroupBox:
        grp  = QGroupBox("Model & Device")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        for i, (label, flag, choices, default, tip) in enumerate(WHISPER_COMBOS):
            row, col = i // 2, (i % 2) * 2
            lbl = QLabel(label + ":")
            lbl.setToolTip(tip)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            cb = QComboBox()
            cb.addItems(choices)
            cb.setCurrentText(default)
            cb.setToolTip(tip)
            cb.setMinimumWidth(140)
            grid.addWidget(lbl, row, col)
            grid.addWidget(cb,  row, col + 1)
            self.w.combos[flag] = cb
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grp

    def _transcription_section(self) -> QGroupBox:
        grp  = QGroupBox("Transcription Settings  (tuned for accuracy)")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        col_count = 3

        # Int spinboxes (skip --port — that goes in server section)
        int_params = [p for p in WHISPER_INT_PARAMS if p[1] != "--port"]
        for i, (label, flag, default, mn, mx, step, tip) in enumerate(int_params):
            row, base_col = i // col_count, (i % col_count) * 2
            lbl = QLabel(label + ":")
            lbl.setToolTip(tip)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            spin = QSpinBox()
            spin.setRange(mn, mx); spin.setValue(default); spin.setSingleStep(step)
            spin.setToolTip(tip); spin.setMinimumWidth(80)
            grid.addWidget(lbl,  row, base_col)
            grid.addWidget(spin, row, base_col + 1)
            self.w.int_spins[flag] = spin

        # Float spinboxes — append to same grid after int params
        offset = len(int_params)
        for i, (label, flag, default, mn, mx, step, decimals, tip) in enumerate(WHISPER_FLOAT_PARAMS):
            idx = offset + i
            row, base_col = idx // col_count, (idx % col_count) * 2
            lbl = QLabel(label + ":")
            lbl.setToolTip(tip)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            spin = QDoubleSpinBox()
            spin.setRange(mn, mx); spin.setValue(default); spin.setSingleStep(step)
            spin.setDecimals(decimals); spin.setToolTip(tip); spin.setMinimumWidth(80)
            grid.addWidget(lbl,  row, base_col)
            grid.addWidget(spin, row, base_col + 1)
            self.w.float_spins[flag] = spin

        # Flags row
        flags_row = (offset + len(WHISPER_FLOAT_PARAMS) + col_count - 1) // col_count
        cb_grp = QGroupBox("Flags")
        cb_grp.setStyleSheet(
            f"QGroupBox {{ color: {TEXT_DIM}; border: 1px solid {BORDER}; }}"
        )
        cb_hl  = QHBoxLayout(cb_grp)
        cb_hl.setSpacing(18)
        for label, flag, tip in WHISPER_CHECKBOXES:
            cb = QCheckBox(label)
            cb.setToolTip(tip)
            cb.setChecked(WHISPER_CHECKBOX_DEFAULTS[flag])
            cb_hl.addWidget(cb)
            self.w.checkboxes[flag] = cb
        cb_hl.addStretch()
        grid.addWidget(cb_grp, flags_row, 0, 1, col_count * 2)
        return grp

    def _server_section(self) -> QGroupBox:
        grp  = QGroupBox("Server")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        # Host + Port on one row
        host_lbl = QLabel("Host:")
        host_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        host_le = QLineEdit("127.0.0.1")
        host_le.setToolTip("Bind address. 127.0.0.1 = localhost only.")
        self.w.string_inputs["--host"] = host_le

        port_lbl = QLabel("Port:")
        port_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        port_spin = QSpinBox()
        port_spin.setRange(1024, 65535); port_spin.setValue(8000); port_spin.setMinimumWidth(80)
        port_spin.setToolTip("HTTP port (default 8000).")
        self.w.int_spins["--port"] = port_spin

        grid.addWidget(host_lbl,  0, 0)
        grid.addWidget(host_le,   0, 1)
        grid.addWidget(port_lbl,  0, 2)
        grid.addWidget(port_spin, 0, 3)

        # Output dir
        out_lbl = QLabel("Output Dir:")
        out_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        out_le = QLineEdit(DEFAULT_WHISPER_OUTPUT)
        out_le.setToolTip("Directory where transcripts are saved.")
        self.w.string_inputs["--output-dir"] = out_le
        grid.addWidget(out_lbl, 1, 0)
        grid.addWidget(out_le,  1, 1, 1, 3)

        # Python interpreter
        py_lbl = QLabel("Python:")
        py_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.python_edit = QLineEdit(sys.executable)
        self.python_edit.setToolTip(
            "Python interpreter. Should be the env where faster-whisper, "
            "fastapi, and uvicorn are installed."
        )
        grid.addWidget(py_lbl,           2, 0)
        grid.addWidget(self.python_edit, 2, 1, 1, 3)

        # Script path
        script_lbl = QLabel("Script Path:")
        script_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        default_script = (str(DEFAULT_WHISPER_SCRIPT)
                          if DEFAULT_WHISPER_SCRIPT.exists() else "whisper_server.py")
        self.script_edit = QLineEdit(default_script)
        self.script_edit.setToolTip("Path to whisper_server.py.")
        grid.addWidget(script_lbl,       3, 0)
        grid.addWidget(self.script_edit, 3, 1, 1, 3)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 0)
        return grp

    # --- command building ---------------------------------------------------
    def _build_command(self) -> list[str]:
        python = self.python_edit.text().strip()
        script = self.script_edit.text().strip()
        if not python or not script:
            return []
        cmd = [python, "-u", script]  # -u for unbuffered output → live log

        # Combos (model/device/compute-type/language)
        for flag, combo in self.w.combos.items():
            val = combo.currentText().strip()
            if val:
                cmd += [flag, val]

        # Strings (host, output-dir)
        for flag, le in self.w.string_inputs.items():
            v = le.text().strip()
            if v:
                cmd += [flag, v]

        # Ints
        for flag, spin in self.w.int_spins.items():
            cmd += [flag, str(spin.value())]

        # Floats
        for flag, spin in self.w.float_spins.items():
            cmd += [flag, f"{spin.value():g}"]

        # Boolean flags — explicit on/off so server defaults can't drift
        for flag, cb in self.w.checkboxes.items():
            if cb.isChecked():
                cmd.append(flag)
            else:
                neg = WHISPER_CHECKBOX_NEGATIVE.get(flag)
                if neg:
                    cmd.append(neg)
        return cmd

    def _validate_before_start(self) -> Optional[str]:
        python = self.python_edit.text().strip()
        script = self.script_edit.text().strip()
        if not python:
            return "Python interpreter path is empty."
        if not script:
            return "whisper_server.py path is empty."
        if not Path(script).is_file():
            return f"Cannot find script:\n{script}"
        return None

    def _open_browser_url(self) -> str:
        host = self.w.string_inputs.get("--host")
        port = self.w.int_spins.get("--port")
        h = (host.text().strip() if host else "") or "127.0.0.1"
        if h == "0.0.0.0":
            h = "127.0.0.1"
        p = port.value() if port else 8000
        # FastAPI's auto-generated docs
        return f"http://{h}:{p}/docs"

    def _wire_preview_signals(self):
        for s in self.w.int_spins.values():
            s.valueChanged.connect(self._update_preview)
        for s in self.w.float_spins.values():
            s.valueChanged.connect(self._update_preview)
        for cb in self.w.checkboxes.values():
            cb.stateChanged.connect(self._update_preview)
        for combo in self.w.combos.values():
            combo.currentIndexChanged.connect(self._update_preview)
        for le in self.w.string_inputs.values():
            le.textChanged.connect(self._update_preview)
        self.python_edit.textChanged.connect(self._update_preview)
        self.script_edit.textChanged.connect(self._update_preview)


# ─── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦙🎙️  AI Server Launcher")
        self.setMinimumSize(900, 860)
        self.resize(1020, 960)
        self.setStyleSheet(STYLESHEET)

        self.tabs = QTabWidget()
        self.llama_tab   = LlamaServerWidget()
        self.whisper_tab = WhisperServerWidget()
        self.tabs.addTab(self.llama_tab,   "🦙  llama.cpp")
        self.tabs.addTab(self.whisper_tab, "🎙️  Whisper")
        self.setCentralWidget(self.tabs)

        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("Ready — pick a tab and start a server.")

    def closeEvent(self, event):
        running = [t for t in (self.llama_tab, self.whisper_tab) if t.is_running()]
        if not running:
            event.accept()
            return
        names = ", ".join(t.tab_label for t in running)
        reply = QMessageBox.question(
            self, "Servers Running",
            f"Still running: {names}.\nStop and exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for t in running:
                t._stop_server()
            event.accept()
        else:
            event.ignore()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ai-server-launcher")
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()