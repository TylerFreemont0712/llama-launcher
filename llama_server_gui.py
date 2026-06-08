#!/usr/bin/env python3
"""
llama_server_gui.py
A simple PyQt6 GUI to launch and monitor a llama.cpp server instance.
Models are loaded from ~/ai/models, server binary from ~/llama.cpp/.
"""

import os
import sys
import shutil
import webbrowser
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QComboBox, QSpinBox,
    QCheckBox, QLineEdit, QPushButton, QTextEdit, QSizePolicy,
    QFrame, QMessageBox, QStatusBar,
)
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QFont, QColor, QTextCursor

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
QSpinBox, QLineEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_MAIN};
    min-height: 26px;
}}
QSpinBox::up-button, QSpinBox::down-button {{
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
"""

# ─── Parameters ───────────────────────────────────────────────────────────────

PARAMS = [
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
    ("Parallel Slots", "--parallel",   1,    1,    64,     1,
     "Simultaneous request slots.\n"
     "Increase only for concurrent API clients."),
    ("Port",           "--port",       8080, 1024, 65535,  1,
     "HTTP port the server listens on.\nDefault: 8080."),
]

CHECKBOXES = [
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

STRING_PARAMS = [
    ("--host", "Host",       "127.0.0.1",
     "IP to bind to.\n127.0.0.1 = local only.  0.0.0.0 = all interfaces."),
    (None,     "Extra Args", "",
     "Extra raw arguments to append.\nExample: --rope-scale 2.0 --log-disable"),
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_server_binary():
    for p in LLAMA_SERVER_CANDIDATES:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which("llama-server") or shutil.which("server") or ""


def scan_models():
    if not MODELS_DIR.exists():
        return []
    found = []
    for pat in ("**/*.gguf", "**/*.GGUF"):
        found.extend(MODELS_DIR.glob(pat))
    return sorted(set(found))


# ─── Main Window ──────────────────────────────────────────────────────────────

class LlamaServerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.process = None

        self.setWindowTitle("🦙  llama.cpp Server Launcher")
        self.setMinimumSize(860, 820)
        self.resize(980, 920)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._refresh_models()
        self._update_buttons()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._header())
        root.addWidget(self._model_section())
        root.addWidget(self._params_section())
        root.addWidget(self._server_section())
        root.addWidget(self._action_bar())
        root.addWidget(self._log_section(), stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — select a model and hit Start Server.")

    def _header(self):
        w  = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("🦙 llama.cpp  Server Launcher")
        title.setFont(QFont("JetBrains Mono", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {HIGHLIGHT};")
        hl.addWidget(title)
        hl.addStretch()
        info = QLabel(f"models: {MODELS_DIR}  |  server: {LLAMA_DIR}")
        info.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        hl.addWidget(info)
        return w

    def _model_section(self):
        grp = QGroupBox("Model Selection")
        hl  = QHBoxLayout(grp)
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_combo.setToolTip("GGUF model files found under ~/ai/models/")
        hl.addWidget(self.model_combo)
        ref = QPushButton("⟳ Refresh")
        ref.setObjectName("refreshBtn")
        ref.clicked.connect(self._refresh_models)
        hl.addWidget(ref)
        return grp

    def _params_section(self):
        grp  = QGroupBox("Server Parameters")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        self.param_widgets = {}
        col_count = 3

        for i, (label, flag, default, mn, mx, step, tip) in enumerate(PARAMS):
            row, base_col = i // col_count, (i % col_count) * 3
            lbl = QLabel(label + ":")
            lbl.setToolTip(tip)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            spin = QSpinBox()
            spin.setRange(mn, mx)
            spin.setValue(default)
            spin.setSingleStep(step)
            spin.setToolTip(tip)
            spin.setMinimumWidth(90)
            grid.addWidget(lbl,  row, base_col)
            grid.addWidget(spin, row, base_col + 1)
            if base_col + 2 < col_count * 3 - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet(f"color: {BORDER};")
                grid.addWidget(sep, row, base_col + 2)
            self.param_widgets[flag] = spin

        cb_row = (len(PARAMS) + col_count - 1) // col_count
        cb_grp = QGroupBox("Flags")
        cb_grp.setStyleSheet(f"QGroupBox {{ color: {TEXT_DIM}; border: 1px solid {BORDER}; }}")
        cb_hl  = QHBoxLayout(cb_grp)
        cb_hl.setSpacing(18)
        self.checkbox_widgets = {}
        for label, flag, tip in CHECKBOXES:
            cb = QCheckBox(label)
            cb.setToolTip(tip)
            cb_hl.addWidget(cb)
            self.checkbox_widgets[flag] = cb
        cb_hl.addStretch()
        grid.addWidget(cb_grp, cb_row, 0, 1, col_count * 3)
        return grp

    def _server_section(self):
        grp  = QGroupBox("Connection & Extra")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.str_widgets = {}

        for row, (flag, label, default, tip) in enumerate(STRING_PARAMS):
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
        self.binary_path_edit = QLineEdit(find_server_binary())
        grid.addWidget(lbl_bin,               len(STRING_PARAMS), 0)
        grid.addWidget(self.binary_path_edit, len(STRING_PARAMS), 1, 1, 3)
        grid.setColumnStretch(1, 1)
        return grp

    def _action_bar(self):
        w  = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        self.cmd_preview = QLabel("")
        self.cmd_preview.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.cmd_preview.setWordWrap(True)
        self.cmd_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        hl.addWidget(self.cmd_preview, stretch=1)

        copy_btn = QPushButton("⎘ Copy")
        copy_btn.setObjectName("copyBtn")
        copy_btn.setToolTip("Copy full server command to clipboard")
        copy_btn.clicked.connect(self._copy_command)
        hl.addWidget(copy_btn)

        self.browser_btn = QPushButton("⌁ Open UI")
        self.browser_btn.setObjectName("browserBtn")
        self.browser_btn.setToolTip("Open the llama.cpp web UI in your browser")
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

        # Wire everything to preview
        for w2 in self.param_widgets.values():
            w2.valueChanged.connect(self._update_preview)
        for w2 in self.checkbox_widgets.values():
            w2.stateChanged.connect(self._update_preview)
        for w2 in self.str_widgets.values():
            w2.textChanged.connect(self._update_preview)
        self.binary_path_edit.textChanged.connect(self._update_preview)
        self.model_combo.currentIndexChanged.connect(self._update_preview)

        return w

    def _log_section(self):
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

    # ── Model scanning ────────────────────────────────────────────────────

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

    # ── Command building ───────────────────────────────────────────────────

    def _build_command(self):
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

    def _update_preview(self):
        cmd = self._build_command()
        self.cmd_preview.setText(
            f"<span style='color:{TEXT_DIM}'>$ {' '.join(cmd)}</span>" if cmd else ""
        )

    # ── Server control ─────────────────────────────────────────────────────

    def _start_server(self):
        cmd = self._build_command()
        if not cmd:
            QMessageBox.warning(self, "Missing Binary",
                                "Server binary path is empty.\nSet it in Connection & Extra.")
            return
        if not self.model_combo.currentData():
            QMessageBox.warning(self, "No Model", "No model selected.")
            return
        if not os.path.isfile(cmd[0]):
            QMessageBox.warning(self, "Binary Not Found",
                                f"Cannot find:\n{cmd[0]}\n\nBuild llama.cpp first.")
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
            self._update_buttons()
            pid = self.process.processId()
            self.status_bar.showMessage(f"Server running — PID {pid}")
            self.log_status.setText(f"● Running  (PID {pid})")
            self.log_status.setStyleSheet("color: #44ff88; font-weight: bold;")

    def _stop_server(self):
        if self.process:
            self._log("\n[Stopping server…]\n", color="#e9c46a")
            self.process.terminate()
            if not self.process.waitForFinished(4000):
                self.process.kill()
            self.process = None
        self._update_buttons()

    def _on_output(self):
        if self.process:
            self._log(bytes(self.process.readAll()).decode("utf-8", errors="replace"))

    def _on_process_finished(self, exit_code, _):
        self._log(f"\n[Process exited — code {exit_code}]\n", color="#e9c46a")
        self.process = None
        self._update_buttons()
        self.status_bar.showMessage(f"Server stopped (exit code {exit_code})")
        self.log_status.setText("● Stopped")
        self.log_status.setStyleSheet(f"color: {TEXT_DIM};")

    def _on_process_error(self, error):
        labels = {
            QProcess.ProcessError.FailedToStart: "FailedToStart",
            QProcess.ProcessError.Crashed:       "Crashed",
            QProcess.ProcessError.Timedout:      "Timedout",
        }
        self._log(f"\n[Process error: {labels.get(error, 'UnknownError')}]\n", color=HIGHLIGHT)
        self.process = None
        self._update_buttons()

    def _update_buttons(self):
        running = (self.process is not None and
                   self.process.state() != QProcess.ProcessState.NotRunning)
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.browser_btn.setEnabled(running)

    def _copy_command(self):
        cmd = self._build_command()
        if cmd:
            QApplication.clipboard().setText(" ".join(cmd))
            self.status_bar.showMessage("Command copied to clipboard.", 3000)

    def _open_browser(self):
        host = self.str_widgets.get("--host")
        port = self.param_widgets.get("--port")
        h = (host.text().strip() if host else "") or "127.0.0.1"
        if h == "0.0.0.0":
            h = "127.0.0.1"
        webbrowser.open(f"http://{h}:{port.value() if port else 8080}")

    # ── Log ───────────────────────────────────────────────────────────────

    def _log(self, text, color=None):
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
        self.log_edit.clear()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            reply = QMessageBox.question(
                self, "Server Running",
                "The server is still running.\nStop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_server()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("llama-server-launcher")
    app.setStyle("Fusion")
    w = LlamaServerGUI()
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()