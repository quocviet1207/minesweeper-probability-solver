"""Preferences: persistence, the dialog, and their effect on the running app."""
import time

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from minesweeper_solver.capture import Region
from minesweeper_solver.capture.controller import CaptureController
from minesweeper_solver.ui import theme
from minesweeper_solver.ui.main_window import MainWindow
from minesweeper_solver.ui.preferences import EFFORT_BUDGETS, Preferences
from minesweeper_solver.ui.settings_dialog import SettingsDialog


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)


@pytest.fixture
def window(app, settings):
    win = MainWindow(settings)
    yield win
    win.shutdown()
    theme.set_palette("standard")          # module-level state: restore for other tests
    win.deleteLater()


CUSTOM = Preferences(interval_ms=350, wait_to_settle=False, hold_frames=5, effort="thorough",
                     prior_density=0.15, decimals=1, highlight_changes=False,
                     highlight_seconds=3.0, palette="colourblind")


def test_preferences_survive_a_restart(settings):
    assert Preferences.load(settings) == Preferences()          # defaults when nothing saved
    CUSTOM.save(settings)
    assert Preferences.load(settings) == CUSTOM


def test_bad_values_in_the_settings_file_are_clamped(settings):
    for key, value in (("interval_ms", "5"), ("hold_frames", "999"), ("effort", "ludicrous"),
                       ("prior_density", "abc"), ("decimals", "7"), ("palette", "neon")):
        settings.setValue(f"prefs/{key}", value)
    prefs = Preferences.load(settings)
    assert prefs.interval_ms == 20 and prefs.hold_frames == 20 and prefs.effort == "normal"
    assert prefs.prior_density == 0.20 and prefs.decimals == 2 and prefs.palette == "standard"


def test_dialog_round_trip(app):
    dialog = SettingsDialog(CUSTOM)
    assert dialog.preferences() == CUSTOM
    dialog.set_preferences(Preferences())
    assert dialog.preferences() == Preferences()


def test_applying_preferences_reaches_every_part(app, window, settings):
    window.apply_preferences(CUSTOM)
    assert window.capture_panel.interval.value() == 350
    assert window.capture_panel.controller.settle is False
    assert window.sync.hold_frames == 5
    assert window.analyser.solve_options == {"node_budget": EFFORT_BUDGETS["thorough"],
                                             "prior_density": 0.15}
    assert window.board_widget.decimals == 1 and window.board_widget.flash_seconds == 0
    window.board_widget.flash([(0, 0)])                          # highlight switched off
    assert not window.board_widget._flashes
    assert theme.risk_color(0.0).name() == theme.PALETTES["colourblind"][0]
    assert window.legend_swatches[0].styleSheet().count(theme.PALETTES["colourblind"][0]) == 1
    assert Preferences.load(settings) == CUSTOM                  # and it was saved


def test_window_starts_with_saved_preferences(app, settings):
    CUSTOM.save(settings)
    win = MainWindow(settings)
    try:
        assert win.prefs == CUSTOM and win.sync.hold_frames == 5
        assert theme.risk_color(1.0).name() == theme.PALETTES["colourblind"][-1]
    finally:
        win.shutdown()
        theme.set_palette("standard")


def test_percentages_with_decimals():
    assert theme.format_percent(0.123) == "12%"
    assert theme.format_percent(0.123, 1) == "12.3%"
    assert theme.format_percent(0.0004, 1) == "<0.1%"
    assert theme.format_percent(0.9996, 1) == ">99.9%"
    assert theme.format_percent(0.0) == "0%" and theme.format_percent(1.0, 2) == "100%"


def test_capture_without_waiting_reads_every_changed_picture(app):
    class Animated:
        def __init__(self):
            self.calls = 0

        def grab(self, region):
            self.calls += 1
            return np.full((region.height, region.width, 3), min(self.calls, 6), np.uint8)

        def close(self):
            pass

    controller = CaptureController(grabber_factory=Animated)
    controller.set_settle(False)
    settled = []
    controller.frameSettled.connect(settled.append)
    controller.set_interval(20)
    controller.set_region(Region(0, 0, 8, 8))
    controller.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and len(settled) < 6:
        app.processEvents()
        time.sleep(0.005)
    time.sleep(0.1)
    app.processEvents()
    controller.stop()
    assert [int(f[0, 0, 0]) for f in settled] == [1, 2, 3, 4, 5, 6]
