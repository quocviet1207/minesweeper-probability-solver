"""Settings dialog: capture, live sync, solver and display preferences."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QLabel, QPushButton, QSpinBox, QVBoxLayout)

from ..vision import glyphs

from .preferences import Preferences


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "muted")
    label.setWordWrap(True)
    return label


class SettingsDialog(QDialog):
    def __init__(self, prefs: Preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        capture = QGroupBox("Capture")
        form = QFormLayout(capture)
        self.interval = QSpinBox(minimum=20, maximum=5000, singleStep=50, suffix=" ms")
        form.addRow("Capture every", self.interval)
        self.settle = QCheckBox("Wait until the picture holds still before reading it")
        form.addRow(self.settle)
        form.addRow(_note("Waiting skips half-finished animations and scrolling, at the cost "
                          "of one capture interval of delay."))
        layout.addWidget(capture)

        sync = QGroupBox("Live sync")
        form = QFormLayout(sync)
        self.hold = QSpinBox(minimum=1, maximum=20, suffix=" frames")
        form.addRow("Keep a revealed number for", self.hold)
        form.addRow(_note("A revealed cell that suddenly reads differently (hover effect, a "
                          "dialog over the board) keeps its number until the new reading has "
                          "been seen this many times."))
        layout.addWidget(sync)

        teach = QGroupBox("Digit recognition")
        form = QFormLayout(teach)
        self.taught = QLabel()
        self.forget = QPushButton("Forget taught digits")
        self.forget.clicked.connect(self._forget)
        form.addRow(self.taught, self.forget)
        form.addRow(_note("While reading from the screen, select a cell shown as ? (or read "
                          "wrongly) and type its number to teach the app that shape."))
        layout.addWidget(teach)
        self._refresh_taught()

        solver = QGroupBox("Solver")
        form = QFormLayout(solver)
        self.effort = QComboBox()
        self.effort.addItem("Fast — estimate large regions sooner", "fast")
        self.effort.addItem("Normal", "normal")
        self.effort.addItem("Thorough — enumerate more before estimating", "thorough")
        form.addRow("Effort", self.effort)
        self.density = QDoubleSpinBox(minimum=1.0, maximum=90.0, decimals=1, singleStep=1.0, suffix=" %")
        form.addRow("Mine density if total unknown", self.density)
        layout.addWidget(solver)

        auto = QGroupBox("Auto-play")
        form = QFormLayout(auto)
        self.auto_flag = QCheckBox("Flag proven mines too")
        form.addRow(self.auto_flag)
        self.auto_guess = QCheckBox("Guess when no safe move is left (the game can be lost)")
        form.addRow(self.auto_guess)
        self.auto_delay = QSpinBox(minimum=0, maximum=3000, singleStep=50, suffix=" ms")
        form.addRow("Pause before each click", self.auto_delay)
        form.addRow(_note("Auto-play only acts after the screen has shown the result of its "
                          "previous click. Moving the mouse hands control back to you at once."))
        layout.addWidget(auto)

        display = QGroupBox("Display")
        form = QFormLayout(display)
        self.decimals = QSpinBox(minimum=0, maximum=2)
        form.addRow("Decimals on the board", self.decimals)
        self.highlight = QCheckBox("Highlight cells that just changed in the live game")
        form.addRow(self.highlight)
        self.highlight_seconds = QDoubleSpinBox(minimum=0.2, maximum=10.0, decimals=1,
                                                singleStep=0.2, suffix=" s")
        form.addRow("Highlight for", self.highlight_seconds)
        self.highlight.toggled.connect(self.highlight_seconds.setEnabled)
        self.palette = QComboBox()
        self.palette.addItem("Green → red", "standard")
        self.palette.addItem("Blue → red (colour-blind friendly)", "colourblind")
        form.addRow("Risk colours", self.palette)
        layout.addWidget(display)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel
                                   | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            lambda: self.set_preferences(Preferences()))
        layout.addWidget(buttons)
        self.set_preferences(prefs)

    def _refresh_taught(self) -> None:
        count = glyphs.user_sample_count() if glyphs._user_file is not None else 0
        self.taught.setText(f"{count} taught shape{'s' if count != 1 else ''}")
        self.forget.setEnabled(count > 0)

    def _forget(self) -> None:
        glyphs.clear_user_samples()
        self._refresh_taught()

    def set_preferences(self, prefs: Preferences) -> None:
        self.interval.setValue(prefs.interval_ms)
        self.settle.setChecked(prefs.wait_to_settle)
        self.hold.setValue(prefs.hold_frames)
        self.effort.setCurrentIndex(max(0, self.effort.findData(prefs.effort)))
        self.density.setValue(prefs.prior_density * 100)
        self.decimals.setValue(prefs.decimals)
        self.highlight.setChecked(prefs.highlight_changes)
        self.highlight_seconds.setValue(prefs.highlight_seconds)
        self.highlight_seconds.setEnabled(prefs.highlight_changes)
        self.palette.setCurrentIndex(max(0, self.palette.findData(prefs.palette)))
        self.auto_flag.setChecked(prefs.autoplay_flag)
        self.auto_guess.setChecked(prefs.autoplay_guess)
        self.auto_delay.setValue(prefs.autoplay_delay_ms)

    def preferences(self) -> Preferences:
        return Preferences(
            interval_ms=self.interval.value(),
            wait_to_settle=self.settle.isChecked(),
            hold_frames=self.hold.value(),
            effort=self.effort.currentData(),
            prior_density=round(self.density.value() / 100, 4),
            decimals=self.decimals.value(),
            highlight_changes=self.highlight.isChecked(),
            highlight_seconds=self.highlight_seconds.value(),
            palette=self.palette.currentData(),
            autoplay_guess=self.auto_guess.isChecked(),
            autoplay_flag=self.auto_flag.isChecked(),
            autoplay_delay_ms=self.auto_delay.value(),
        ).validated()
