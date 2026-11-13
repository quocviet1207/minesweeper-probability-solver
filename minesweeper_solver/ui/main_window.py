"""Main window: capture panel (left), board (centre), solver analysis (right)."""
from __future__ import annotations

import time
from html import escape as html_escape
from typing import Optional

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget)

from ..core import (ESTIMATED, EXACT, FLAG, MINE, PRIOR, UNCERTAIN, UNKNOWN, UNREVEALED, Board,
                    SolverCache, cell_name, explain_cell, recommend, solve)
from ..state import CHANGED, NEW_GAME, UNCHANGED, GameStateManager
from ..core.board import Cell, random_position
from . import theme
from .autoplay_controller import AutoPlayController
from .capture_panel import CapturePanel
from .flow_layout import FlowLayout
from ..vision import glyphs
from .preferences import Preferences
from .recognition_worker import Analysis, AnalysisWorker, RecognitionWorker
from .settings_dialog import SettingsDialog
from .board_widget import (TOOL_SELECT, VIEW_MINE, VIEW_NORMAL, VIEW_RECOMMEND, VIEW_SAFE,
                           BoardWidget)

PRESETS = {
    "Beginner  9×9 · 10": (9, 9, 10),
    "Intermediate  16×16 · 40": (16, 16, 40),
    "Expert  16×30 · 99": (16, 30, 99),
}
QUALITY_TEXT = {
    EXACT: "Exact",
    ESTIMATED: "Estimated (sampled — region too large to enumerate)",
    PRIOR: "Depends on assumed mine density (total unknown)",
    UNCERTAIN: "Approximate (some cells were not recognised)",
}
STATE_TEXT = {UNREVEALED: "Covered", FLAG: "Flagged", MINE: "Exploded mine", UNKNOWN: "Not recognised"}


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "muted")
    label.setWordWrap(True)
    return label


def _wrapped(role: Optional[str] = None) -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if role:
        label.setProperty("role", role)
    return label


class MainWindow(QMainWindow):
    def __init__(self, settings: Optional[QSettings] = None, taught_digits: Optional[str] = None,
                 mouse=None):
        super().__init__()
        glyphs.set_user_samples_file(taught_digits)
        self._last_recognition = None
        self.setWindowTitle("Minesweeper Solver & Probability Analyzer")
        self.settings = settings
        self.board = random_position(9, 9, 10, seed=4)
        self.result = None
        self.recommendation = None
        self.sync = GameStateManager()
        self.worker = RecognitionWorker(self)
        self.worker.resultReady.connect(self._on_recognition)
        self.worker.start()
        self.analyser = AnalysisWorker(self)
        self.analyser.resultReady.connect(self._on_analysis)
        self.analyser.start()
        self.solver_cache = self.analyser.cache
        self._ticket = 0
        self._analysed_board: Optional[Board] = None
        self._sync_text = ""
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        self._busy_timer.setInterval(150)
        self._busy_timer.timeout.connect(lambda: self._set_busy(True))
        app = QApplication.instance()
        if app is not None:          # also when the app quits without closing this window
            app.aboutToQuit.connect(self.shutdown)
        self._last_solve_ms = 0.0

        self.board_widget = BoardWidget(self.board)
        self.board_widget.boardEdited.connect(self._schedule_solve)
        self.board_widget.digitTyped.connect(self._teach_digit)
        self.board_widget.selectionChanged.connect(lambda _c: self._refresh_selection())

        self._solve_timer = QTimer(self)
        self._solve_timer.setSingleShot(True)
        self._solve_timer.setInterval(30)
        self._solve_timer.timeout.connect(self._solve)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 6, 8, 6)
        # three panes with draggable dividers; either side pane can be dragged
        # shut or hidden with the buttons above the board
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setHandleWidth(8)
        self.left_pane = self._scroll(self._build_left())
        self.right_pane = self._scroll(self._build_right())
        center = self._build_center()
        center.setMinimumWidth(220)
        for pane in (self.left_pane, center, self.right_pane):
            self.splitter.addWidget(pane)
        self.splitter.setCollapsible(1, False)
        for index, stretch in enumerate((1, 3, 1)):    # shrink sides too, board most
            self.splitter.setStretchFactor(index, stretch)
        self.splitter.setSizes([320, 700, 340])
        self.splitter.splitterMoved.connect(lambda *_: self._sync_pane_buttons())
        layout.addWidget(self.splitter)
        self.setCentralWidget(central)
        self._restore_window()
        self.statusBar().showMessage("Select a capture region to read a game from the screen, or edit the board by hand.")
        self._sync_setup_controls()
        self.prefs = Preferences.load(settings)
        self.autoplay = AutoPlayController(self, mouse)
        self.autoplay.runningChanged.connect(self._autoplay_changed)
        self.autoplay.status.connect(self._autoplay_status)
        self.autoplay_button.setEnabled(self.autoplay.available)
        if not self.autoplay.available:
            self.autoplay_button.setToolTip("Mouse control is not available on this system.")
        self.capture_panel.controller.stateChanged.connect(
            lambda state: state != "running" and self.autoplay.stop("Capture stopped, so auto-play stopped."))
        self.apply_preferences(self.prefs, save=False)

    def closeEvent(self, event):
        self._save_window()
        self.shutdown()
        super().closeEvent(event)

    # ----------------------------------------------------------- window size
    def _restore_window(self) -> None:
        """Last size and pane widths if saved, otherwise a size that fits the screen."""
        restored = False
        if self.settings is not None:
            geometry = self.settings.value("window/geometry")
            panes = self.settings.value("window/panes")
            if isinstance(geometry, QByteArray) and not geometry.isEmpty():
                restored = self.restoreGeometry(geometry)
            if isinstance(panes, QByteArray) and not panes.isEmpty():
                self.splitter.restoreState(panes)
            for toggle, key in ((self.left_toggle, "window/left_pane"), (self.right_toggle, "window/right_pane")):
                if str(self.settings.value(key, "true")).lower() == "false":
                    toggle.setChecked(False)           # hides the pane
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self._sync_pane_buttons()
            return
        area = screen.availableGeometry()
        if not restored or not area.intersects(self.frameGeometry()):
            width = min(1360, area.width() - 40)
            height = min(820, area.height() - 40)
            self.resize(width, height)
            self.move(area.x() + (area.width() - width) // 2, area.y() + (area.height() - height) // 2)
        self._sync_pane_buttons()

    def _save_window(self) -> None:
        if self.settings is not None:
            self.settings.setValue("window/geometry", self.saveGeometry())
            self.settings.setValue("window/panes", self.splitter.saveState())
            self.settings.setValue("window/left_pane", not self.left_pane.isHidden())
            self.settings.setValue("window/right_pane", not self.right_pane.isHidden())

    def _show_pane(self, index: int, visible: bool) -> None:
        pane = self.splitter.widget(index)
        pane.setVisible(visible)
        sizes = self.splitter.sizes()
        if visible and sizes[index] == 0:          # reopen a pane that was dragged shut
            sizes[index] = 320 if index == 0 else 340
            self.splitter.setSizes(sizes)

    def _sync_pane_buttons(self) -> None:
        """Keep the toggle buttons right when a pane is dragged shut or open."""
        sizes = self.splitter.sizes()
        for button, index in ((self.left_toggle, 0), (self.right_toggle, 2)):
            shown = not self.splitter.widget(index).isHidden() and sizes[index] > 0
            button.blockSignals(True)
            button.setChecked(shown)
            button.blockSignals(False)

    def shutdown(self) -> None:
        """Stop the capture, recognition and analysis threads (safe to call twice)."""
        if hasattr(self, "autoplay"):
            self.autoplay.stop("Closing.")
        self.capture_panel.shutdown()
        for worker in (self.worker, self.analyser):
            if worker.isRunning():
                worker.stop()

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        """Side pane that can be made narrower than its contents (they scroll)."""
        area = QScrollArea()
        area.setWidget(widget)
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setMinimumWidth(160)
        return area

    # ------------------------------------------------------------------ left
    def _build_left(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 6, 0)

        self.capture_panel = CapturePanel(settings=self.settings)
        self.capture_panel.message.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.capture_panel.frameSettled.connect(self._on_frame)
        self.capture_panel.read_board.toggled.connect(self._reading_toggled)
        col.addWidget(self.capture_panel)

        setup = QGroupBox("Board setup")
        form = QFormLayout(setup)
        self.preset = QComboBox()
        self.preset.addItems(["Custom"] + list(PRESETS))
        self.preset.activated.connect(self._apply_preset)
        form.addRow("Preset", self.preset)
        self.rows_spin = QSpinBox(minimum=1, maximum=60)
        self.cols_spin = QSpinBox(minimum=1, maximum=80)
        self.rows_spin.valueChanged.connect(self._resize_board)
        self.cols_spin.valueChanged.connect(self._resize_board)
        form.addRow("Rows", self.rows_spin)
        form.addRow("Columns", self.cols_spin)
        self.mines_known = QCheckBox("Total mines known")
        self.mines_spin = QSpinBox(minimum=0, maximum=4800)
        self.mines_known.toggled.connect(self._mines_changed)
        self.mines_spin.valueChanged.connect(self._mines_changed)
        form.addRow(self.mines_known)
        form.addRow("Total mines", self.mines_spin)
        clear = QPushButton("Clear board")
        clear.clicked.connect(self._clear_board)
        example = QPushButton("Random example")
        example.clicked.connect(self._load_example)
        buttons = QHBoxLayout()
        buttons.addWidget(clear)
        buttons.addWidget(example)
        form.addRow(buttons)
        col.addWidget(setup)

        tools = QGroupBox("Edit tool")
        grid = QGridLayout(tools)
        grid.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        entries = [("Select", TOOL_SELECT), ("Covered", UNREVEALED), ("⚑ Flag", FLAG)]
        entries += [(str(n), n) for n in range(9)] + [("✹ Mine", MINE)]
        for i, (text, tool) in enumerate(entries):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, t=tool: setattr(self.board_widget, "tool", t))
            self.tool_group.addButton(b)
            if i < 3:
                grid.addWidget(b, 0, i * 2, 1, 2)
            elif i < 12:
                grid.addWidget(b, 1 + (i - 3) // 6, (i - 3) % 6)
            else:
                grid.addWidget(b, 2, 3, 1, 3)
            if tool == TOOL_SELECT:
                b.setChecked(True)
        grid.addWidget(_muted("Click or drag to paint. Right-click toggles a flag. "
                              "Keyboard on the selected cell: 0–8, F flag, Space covered, arrows move."),
                       3, 0, 1, 6)
        col.addWidget(tools)

        col.addStretch(1)
        return panel

    # ---------------------------------------------------------------- centre
    def _build_center(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 6, 0, 0)

        bar_box = QWidget()
        bar = FlowLayout(bar_box, h_spacing=10, v_spacing=6)    # wraps when narrow
        self.left_toggle = QToolButton()
        self.left_toggle.setText("◧ Capture")
        self.left_toggle.setToolTip("Show or hide the capture and board setup panel")
        self.left_toggle.setCheckable(True)
        self.left_toggle.setChecked(True)
        self.left_toggle.toggled.connect(lambda on: self._show_pane(0, on))
        bar.addWidget(self.left_toggle)
        view_label = _muted("View")
        view_label.setWordWrap(False)
        bar.addWidget(view_label)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Normal board", "Mine probability heatmap",
                                  "Safe probability heatmap", "Solver recommendation"])
        self.view_combo.setCurrentIndex(VIEW_MINE)
        self.view_combo.currentIndexChanged.connect(self.board_widget.set_view)
        bar.addWidget(self.view_combo)
        percent = QCheckBox("Show percentages")
        percent.setChecked(True)
        percent.toggled.connect(self.board_widget.set_show_percent)
        bar.addWidget(percent)
        settings_button = QPushButton("⚙ Settings")
        settings_button.clicked.connect(self.open_settings)
        bar.addWidget(settings_button)
        self.autoplay_button = QToolButton()
        self.autoplay_button.setText("▶ Auto-play")
        self.autoplay_button.setToolTip(
            "Let the app click guaranteed-safe cells in the game for you. "
            "Move the mouse at any time to take over.")
        self.autoplay_button.setCheckable(True)
        self.autoplay_button.toggled.connect(self._autoplay_toggled)
        bar.addWidget(self.autoplay_button)
        self.right_toggle = QToolButton()
        self.right_toggle.setText("Analysis ◨")
        self.right_toggle.setToolTip("Show or hide the analysis panel")
        self.right_toggle.setCheckable(True)
        self.right_toggle.setChecked(True)
        self.right_toggle.toggled.connect(lambda on: self._show_pane(2, on))
        bar.addWidget(self.right_toggle)
        v.addWidget(bar_box)

        self.autoplay_label = QLabel()
        self.autoplay_label.setWordWrap(True)
        self.autoplay_label.hide()
        v.addWidget(self.autoplay_label)
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("background: #5a1d1d; color: #ffd9d9; border-radius: 6px; padding: 8px 10px;")
        self.banner.hide()
        v.addWidget(self.banner)
        v.addWidget(self.board_widget, 1)

        self.legend_box = QWidget()
        v.addWidget(self.legend_box)
        self._build_legend()
        return box

    def _build_legend(self) -> None:
        old = self.legend_box.layout()
        if old is not None:                      # rebuild after a palette change
            while old.count():
                item = old.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            QWidget().setLayout(old)
        legend = FlowLayout(self.legend_box, h_spacing=12, v_spacing=4)   # wraps when narrow
        title = QLabel("Mine risk")
        title.setProperty("role", "muted")
        legend.addWidget(title)
        self.legend_swatches = []
        for _, color, label in theme.RISK_BANDS:
            entry = QWidget()
            row = QHBoxLayout(entry)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(5)
            swatch = QFrame()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background: {color.name()}; border-radius: 3px;")
            self.legend_swatches.append(swatch)
            row.addWidget(swatch)
            text = QLabel(label)
            text.setProperty("role", "muted")
            row.addWidget(text)
            legend.addWidget(entry)
        note = QLabel("~ = not exact")
        note.setProperty("role", "muted")
        legend.addWidget(note)

    # ------------------------------------------------------------- teaching
    def _teach_digit(self, cell, digit: int) -> None:
        """While reading from the screen, typing a digit on a cell that was not
        recognised (or was read wrongly) teaches the app that glyph."""
        result = self._last_recognition
        if not self._reading() or result is None or result.features is None:
            return
        if result.states.shape != (self.board.rows, self.board.cols):
            return
        mask = result.features["masks"][cell]
        if not mask.any() or int(result.states[cell]) == digit:
            return                               # no glyph here, or already read right
        if glyphs._user_file is None:
            self.statusBar().showMessage("Teaching is not available: no place to store taught digits.", 6000)
            return
        added = glyphs.add_user_samples(mask[None], [digit])
        total = glyphs.user_sample_count()
        self.statusBar().showMessage(
            f"Learned: that shape is a {digit}" + ("" if added else " (already known)")
            + f" · {total} taught shape{'s' if total != 1 else ''} · reading the board again", 8000)
        self.worker.reset()                      # re-read the current picture with the new shape
        self.sync.reset()
        frame = self.capture_panel._last_frame
        if frame is not None:
            self.worker.submit(frame)

    # ------------------------------------------------------------- settings
    def open_settings(self) -> None:
        current = Preferences(**{**self.prefs.__dict__,
                                 "interval_ms": self.capture_panel.interval.value()})
        dialog = SettingsDialog(current, self)
        if dialog.exec():
            self.apply_preferences(dialog.preferences())

    def apply_preferences(self, prefs: Preferences, save: bool = True) -> None:
        prefs = prefs.validated()
        self.prefs = prefs
        if self.capture_panel.interval.value() != prefs.interval_ms:
            self.capture_panel.interval.setValue(prefs.interval_ms)
        self.capture_panel.controller.set_settle(prefs.wait_to_settle)
        self.sync.hold_frames = prefs.hold_frames
        self.analyser.solve_options = prefs.solve_options()
        self.board_widget.decimals = prefs.decimals
        self.board_widget.flash_seconds = prefs.highlight_seconds if prefs.highlight_changes else 0.0
        if theme.RISK_BANDS[0][1].name() != theme.PALETTES[prefs.palette][0]:
            theme.set_palette(prefs.palette)
            self._build_legend()
        self.board_widget.update()
        if save:
            prefs.save(self.settings)
        self._solve()                            # solver options may have changed

    # ----------------------------------------------------------------- right
    def _build_right(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 6, 0)

        self.analysis_state = QLabel("")
        self.analysis_state.setProperty("role", "muted")
        self.analysis_state.hide()
        col.addWidget(self.analysis_state)

        stats = QGroupBox("Board status")
        grid = QGridLayout(stats)
        self.stat_labels = {}
        for i, (key, title) in enumerate([
            ("remaining", "Mines remaining"), ("covered", "Covered cells"),
            ("mines", "Confirmed mines"), ("safe", "Guaranteed safe"),
        ]):
            value = QLabel("–")
            value.setProperty("role", "value")
            self.stat_labels[key] = value
            grid.addWidget(value, (i // 2) * 2, i % 2)
            grid.addWidget(_muted(title), (i // 2) * 2 + 1, i % 2)
        col.addWidget(stats)

        rec = QGroupBox("Recommended next move")
        v = QVBoxLayout(rec)
        self.rec_headline = _wrapped("headline")
        self.rec_text = _wrapped()
        self.rec_lists = _wrapped("muted")
        v.addWidget(self.rec_headline)
        v.addWidget(self.rec_text)
        v.addWidget(self.rec_lists)
        col.addWidget(rec)

        sel = QGroupBox("Selected cell")
        v = QVBoxLayout(sel)
        self.sel_headline = _wrapped("headline")
        self.sel_text = _wrapped()
        v.addWidget(self.sel_headline)
        v.addWidget(self.sel_text)
        col.addWidget(sel)

        notes = QGroupBox("Solver reasoning")
        v = QVBoxLayout(notes)
        self.notes_text = _wrapped("muted")
        v.addWidget(self.notes_text)
        col.addWidget(notes)
        col.addStretch(1)
        return panel

    # ------------------------------------------------------------- board ops
    def _sync_setup_controls(self) -> None:
        for w in (self.rows_spin, self.cols_spin, self.mines_spin, self.mines_known):
            w.blockSignals(True)
        self.rows_spin.setValue(self.board.rows)
        self.cols_spin.setValue(self.board.cols)
        self.mines_known.setChecked(self.board.total_mines is not None)
        if self.board.total_mines is not None:
            self.mines_spin.setValue(self.board.total_mines)
        self.mines_spin.setEnabled(self.board.total_mines is not None)
        for w in (self.rows_spin, self.cols_spin, self.mines_spin, self.mines_known):
            w.blockSignals(False)
        key = (self.board.rows, self.board.cols, self.board.total_mines)
        names = [name for name, preset in PRESETS.items() if preset == key]
        self.preset.setCurrentText(names[0] if names else "Custom")

    def _replace_board(self, board: Board) -> None:
        self.board = board
        self.board_widget.set_board(board)
        self._sync_setup_controls()
        self._schedule_solve()

    def _apply_preset(self, index: int) -> None:
        name = self.preset.itemText(index)
        if name in PRESETS:
            rows, cols, mines = PRESETS[name]
            self._replace_board(Board(rows, cols, mines))

    def _resize_board(self) -> None:
        self._replace_board(self.board.resized(self.rows_spin.value(), self.cols_spin.value()))

    def _mines_changed(self) -> None:
        self.board.total_mines = self.mines_spin.value() if self.mines_known.isChecked() else None
        self._sync_setup_controls()
        self._schedule_solve()

    def _clear_board(self) -> None:
        self._replace_board(Board(self.board.rows, self.board.cols, self.board.total_mines))

    def _load_example(self) -> None:
        rows, cols = self.board.rows, self.board.cols
        mines = self.board.total_mines
        if mines is None or not 0 < mines <= rows * cols - 9:
            mines = max(1, round(rows * cols * 0.16))
        self._replace_board(random_position(rows, cols, mines))

    # ----------------------------------------------------------- recognition
    def _reading(self) -> bool:
        return self.capture_panel.read_board.isChecked()

    def _reading_toggled(self, on: bool) -> None:
        if not on:
            self.autoplay.stop("Reading from the screen was switched off, so auto-play stopped.")
        self.worker.reset()
        self.sync.reset()
        frame = self.capture_panel._last_frame
        if on and frame is not None:
            self.worker.submit(frame)           # read the current picture now

    def _on_frame(self, frame) -> None:
        """The captured picture has settled: read it in the background."""
        if self._reading():
            self.worker.submit(frame)

    def _on_recognition(self, result, generation: int) -> None:
        if generation != self.worker.generation or not self._reading():
            return                               # stale: reading was reset meanwhile
        if result.ok:
            self._last_recognition = result
        self.capture_panel.show_recognition(result)
        controller = self.capture_panel.controller
        if not result.ok:
            controller.set_settle_region(None)
            self._show_sync(result, None)
            return
        g = result.grid                          # judge stillness on the board only
        controller.set_settle_region((int(g.y_lines[0]) - 2, int(g.y_lines[-1]) + 3,
                                      int(g.x_lines[0]) - 2, int(g.x_lines[-1]) + 3))
        if result.skipped:
            self._show_sync(result, None)        # no cell changed: nothing to do
            return
        same_size = result.states.shape == (self.board.rows, self.board.cols)
        update = self.sync.apply(result.states, self.board.total_mines if same_size else None)
        if update.event != UNCHANGED:
            self.board = update.board
            self.board_widget.set_board(update.board)
            if update.event == CHANGED:
                self.board_widget.flash(update.changed)
            self._sync_setup_controls()
            self._solve()
        self._show_sync(result, update)

    def _show_sync(self, result, update) -> None:
        if not result.ok:
            text = "Live: no board in the capture region"
        elif result.skipped or update is None:
            text = "Live: picture changed outside the board — nothing to update"
        else:
            text = f"Live: {update.summary()}"
            if update.held:
                text += f" · {len(update.held)} revealed cell(s) kept despite a different reading"
            if update.event == NEW_GAME:
                text += " · board replaced"
            if update.game_over:
                text += " · game over"
        text += f" · read {result.elapsed_ms:.0f} ms"
        if self.worker.dropped:
            text += f" · {self.worker.dropped} stale frame(s) skipped"
        self._sync_text = text
        self.statusBar().showMessage(text)

    # --------------------------------------------------------------- solving
    def _schedule_solve(self) -> None:
        self._solve_timer.start()

    def _solve(self) -> None:
        """Analyse a snapshot of the board in the background."""
        self._ticket += 1
        self.analyser.submit((self._ticket, self.board.copy()))
        if not self._busy_timer.isActive():
            self._busy_timer.start()

    def analyse_now(self) -> None:
        """Analyse synchronously (scripts and tests that need the result at once)."""
        self._ticket += 1
        started = time.perf_counter()
        board = self.board.copy()
        result = solve(board)
        self._on_analysis(Analysis(self._ticket, board, result, recommend(board, result),
                                   (time.perf_counter() - started) * 1000), 0)

    def _set_busy(self, busy: bool) -> None:
        self.analysis_state.setText("Updating analysis…" if busy else "")
        self.analysis_state.setVisible(busy)

    def _on_analysis(self, analysis: Analysis, _generation: int) -> None:
        if analysis.ticket != self._ticket:
            return                                   # a newer board is already queued
        self._busy_timer.stop()
        self._set_busy(False)
        if analysis.error:
            self.banner.setText(f"⚠ {analysis.error}")
            self.banner.show()
            return
        self.result, self.recommendation = analysis.result, analysis.recommendation
        self._analysed_board = analysis.board
        self._last_solve_ms = analysis.elapsed_ms
        self.board_widget.set_analysis(self.result, self.recommendation)
        self._refresh_analysis()
        self._refresh_selection()
        if self._reading() and self._sync_text:
            text = self._sync_text + f" · solved {analysis.elapsed_ms:.0f} ms"
            if analysis.result.cached_components:
                text += (f" ({analysis.result.cached_components} of "
                         f"{analysis.result.components} regions reused)")
            self.statusBar().showMessage(text)
        self.autoplay.tick()

    def analysis_current(self) -> bool:
        """True if the analysis on screen belongs to exactly the board on screen."""
        done = self._analysed_board
        return (done is not None and self.result is not None
                and done.cells == self.board.cells and done.total_mines == self.board.total_mines)

    # -------------------------------------------------------------- auto-play
    def _autoplay_toggled(self, on: bool) -> None:
        if on and not self.autoplay.running:
            if not self.autoplay.start():
                self.autoplay_button.blockSignals(True)
                self.autoplay_button.setChecked(False)
                self.autoplay_button.blockSignals(False)
        elif not on and self.autoplay.running:
            self.autoplay.stop("Auto-play stopped.")

    def _autoplay_changed(self, running: bool) -> None:
        self.autoplay_button.blockSignals(True)
        self.autoplay_button.setChecked(running)
        self.autoplay_button.setText("■ Stop auto-play" if running else "▶ Auto-play")
        self.autoplay_button.blockSignals(False)

    def _autoplay_status(self, text: str, kind: str) -> None:
        colours = {"running": ("#17324d", "#cfe6ff"), "finished": ("#1d4a2c", "#d4f5dd"),
                   "stopped": ("#4a3b17", "#ffe9b8"), "error": ("#5a1d1d", "#ffd9d9")}
        bg, fg = colours.get(kind, colours["stopped"])
        self.autoplay_label.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 6px; padding: 6px 10px;")
        self.autoplay_label.setText(text)
        self.autoplay_label.show()

    def _refresh_analysis(self) -> None:
        res, rec = self.result, self.recommendation
        if res.valid:
            self.banner.hide()
        else:
            self.banner.setText(f"⚠ Inconsistent board — no probabilities shown. {res.error}")
            self.banner.show()

        remaining = "?" if res.remaining_mines is None else str(res.remaining_mines)
        self.stat_labels["remaining"].setText(remaining)
        self.stat_labels["covered"].setText(str(res.unrevealed_count))
        self.stat_labels["mines"].setText(
            str(res.flag_count + len(res.mine_cells)) if res.valid else "–")
        self.stat_labels["safe"].setText(str(len(res.safe_cells)) if res.valid else "–")
        self.stat_labels["mines"].setToolTip(
            f"{res.flag_count} flagged + {len(res.mine_cells)} proven by the solver")

        self.rec_headline.setText(rec.headline)
        colors = {"safe": "#5fd17a", "guess": "#f5a142", "flag": "#ff6b6b", "invalid": "#ff6b6b"}
        self.rec_headline.setStyleSheet(f"color: {colors.get(rec.kind, theme.TEXT)};")
        self.rec_text.setText(rec.explanation)
        lines = []
        if rec.safe_cells:
            lines.append("<b>Safe to reveal:</b> " + self._cell_list(rec.safe_cells))
        if rec.mine_cells:
            lines.append("<b>Proven mines:</b> " + self._cell_list(rec.mine_cells))
        if rec.alternatives:
            lines.append("<b>Next best guesses:</b> " + ", ".join(
                f"{cell_name(c)} ({p * 100:.1f}%)" for c, p in rec.alternatives))
        self.rec_lists.setText("<br>".join(lines))
        self.rec_lists.setVisible(bool(lines))

        self.notes_text.setText("\n\n".join(res.notes) if res.valid and res.notes
                                else ("—" if res.valid else res.error))

    @staticmethod
    def _cell_list(cells: list[Cell], limit: int = 24) -> str:
        text = ", ".join(cell_name(c) for c in cells[:limit])
        return text + (f" … (+{len(cells) - limit})" if len(cells) > limit else "")

    def _refresh_selection(self) -> None:
        cell = self.board_widget.selected
        if cell is None or not self.board.in_bounds(cell):
            self.sel_headline.setText("No cell selected")
            self.sel_text.setText("Click a cell to see its probability and the reasoning behind it.")
            return
        state = self.board[cell]
        name = cell_name(cell)
        res = self.result
        if state >= 0:
            headline = f"{name} — revealed {state}"
        elif state == UNKNOWN and self._reading():
            self.sel_headline.setText(f"{name} — not recognised")
            self.sel_text.setText(
                "This cell could not be read with confidence, so it is not guessed. If it shows "
                "a number, type that number (1–8): the app learns this shape, remembers it, and "
                "reads the board again. Typing a number on a cell that was read wrongly teaches "
                "it the same way.")
            return
        elif state == UNREVEALED and res is not None and res.valid and cell in res.probs:
            approx = "" if res.quality[cell] == EXACT else "≈ "
            headline = f"{name} — {approx}{res.probs[cell] * 100:.1f}% mine"
        else:
            headline = f"{name} — {STATE_TEXT.get(state, 'covered').lower()}"
        self.sel_headline.setText(headline)
        if res is None:
            self.sel_text.setText("Analysis in progress…")
            return
        sections = explain_cell(self.board, res, self.recommendation, cell)
        html = "".join(
            f"<p style='margin:0 0 6px 0'><b>{html_escape(title)}</b><br>"
            + "<br>".join(html_escape(line) for line in text.splitlines()) + "</p>"
            for title, text in sections if text)
        self.sel_text.setText(html)
