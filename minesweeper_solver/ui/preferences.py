"""User preferences, stored with QSettings between runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional

from PySide6.QtCore import QSettings

# solver effort -> work budget for exact enumeration of one board
EFFORT_BUDGETS = {"fast": 500_000, "normal": 3_000_000, "thorough": 15_000_000}


@dataclass
class Preferences:
    # capture
    interval_ms: int = 200
    wait_to_settle: bool = True        # read a picture only once it holds still
    # live sync
    hold_frames: int = 3               # frames a revealed number survives a different reading
    # solver
    effort: str = "normal"             # key of EFFORT_BUDGETS
    prior_density: float = 0.20        # assumed mine density when the total is unknown
    # display
    decimals: int = 0                  # decimals on the board's percentages
    highlight_changes: bool = True
    highlight_seconds: float = 1.6
    palette: str = "standard"          # theme.PALETTES key
    # auto-play
    autoplay_guess: bool = False       # also guess when no safe move is left (can lose)
    autoplay_flag: bool = False        # flag proven mines too
    autoplay_delay_ms: int = 150       # pause before each click

    def solve_options(self) -> dict:
        return {"node_budget": EFFORT_BUDGETS.get(self.effort, EFFORT_BUDGETS["normal"]),
                "prior_density": self.prior_density}

    def validated(self) -> "Preferences":
        """Clamp values read from a settings file that may have been edited by hand."""
        p = Preferences(**asdict(self))
        p.interval_ms = min(max(int(p.interval_ms), 20), 5000)
        p.hold_frames = min(max(int(p.hold_frames), 1), 20)
        p.effort = p.effort if p.effort in EFFORT_BUDGETS else "normal"
        p.prior_density = min(max(float(p.prior_density), 0.01), 0.9)
        p.decimals = min(max(int(p.decimals), 0), 2)
        p.highlight_seconds = min(max(float(p.highlight_seconds), 0.2), 10.0)
        p.palette = p.palette if p.palette in ("standard", "colourblind") else "standard"
        p.autoplay_delay_ms = min(max(int(p.autoplay_delay_ms), 0), 3000)
        return p

    # -- persistence -------------------------------------------------------------
    @classmethod
    def load(cls, settings: Optional[QSettings]) -> "Preferences":
        prefs = cls()
        if settings is None:
            return prefs
        values = {}
        for f in fields(cls):
            raw = settings.value(f"prefs/{f.name}")
            if raw is None:
                continue
            default = getattr(prefs, f.name)
            try:
                if isinstance(default, bool):          # QSettings may hand back "true"/"false"
                    values[f.name] = raw if isinstance(raw, bool) else str(raw).lower() == "true"
                else:
                    values[f.name] = type(default)(raw)
            except (TypeError, ValueError):
                pass
        return cls(**{**asdict(prefs), **values}).validated()

    def save(self, settings: Optional[QSettings]) -> None:
        if settings is None:
            return
        for name, value in asdict(self).items():
            settings.setValue(f"prefs/{name}", value)
        settings.sync()
