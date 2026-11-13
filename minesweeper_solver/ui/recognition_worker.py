"""Background threads that always work on the newest request only.

If requests arrive faster than they can be handled, older waiting requests are
dropped instead of queued, so results never lag behind by more than one job.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core import Board, Recommendation, SolveResult, SolverCache, recommend, solve
from ..vision import BoardReader, Recognition


class LatestOnlyWorker(QThread):
    """Base class: subclasses implement `handle` (and optionally `setup`,
    `on_reset`, `error_result`); everything except `submit`, `reset` and
    `stop` runs on the worker thread."""

    resultReady = Signal(object, int)    # result, generation it belongs to

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._pending: Any = None
        self._has_pending = False
        self._pending_generation = 0
        self._generation = 0
        self._reset = False
        self._stop = False
        self.submitted = 0
        self.dropped = 0
        self.processed = 0

    # -- called from the UI thread -----------------------------------------------
    @property
    def generation(self) -> int:
        """Results from an older generation are stale (see `reset`)."""
        return self._generation

    def submit(self, item: Any) -> None:
        with self._cond:
            if self._has_pending:
                self.dropped += 1
            self._pending, self._has_pending = item, True
            self._pending_generation = self._generation
            self.submitted += 1
            self._cond.notify()

    def reset(self) -> None:
        """Forget accumulated state. Results already in flight keep the old
        generation number so the receiver can ignore them."""
        with self._cond:
            self._generation += 1
            self._pending, self._has_pending = None, False
            self._reset = True

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self.wait(10000)

    # -- worker thread ---------------------------------------------------------------
    def setup(self) -> None:
        pass

    def on_reset(self) -> None:
        pass

    def handle(self, item: Any) -> Any:
        raise NotImplementedError

    def error_result(self, item: Any, exc: Exception) -> Any:
        return exc

    def run(self) -> None:
        self.setup()
        while True:
            with self._cond:
                while not self._stop and not self._has_pending:
                    self._cond.wait()
                if self._stop:
                    return
                item, generation = self._pending, self._pending_generation
                self._pending, self._has_pending = None, False
                reset, self._reset = self._reset, False
            if reset:
                self.on_reset()
            try:
                result = self.handle(item)
            except Exception as exc:          # never let a bad request kill the thread
                result = self.error_result(item, exc)
            self.processed += 1
            self.resultReady.emit(result, generation)


class RecognitionWorker(LatestOnlyWorker):
    """Reads boards from captured frames (keeps a BoardReader between frames)."""

    def __init__(self, parent=None, reader_factory: Callable[[], BoardReader] = BoardReader):
        super().__init__(parent)
        self._reader_factory = reader_factory
        self.reader: Optional[BoardReader] = None

    def setup(self) -> None:
        self.reader = self._reader_factory()
        try:                                  # load digit templates before the first frame
            from ..vision.glyphs import load_templates
            load_templates()
        except Exception:
            pass

    def on_reset(self) -> None:
        self.reader.reset()

    def handle(self, frame: np.ndarray) -> Recognition:
        return self.reader.read(frame)

    def error_result(self, frame, exc: Exception) -> Recognition:
        self.reader.reset()
        return Recognition(False, f"Recognition failed: {exc}")


@dataclass
class Analysis:
    ticket: int                      # matches the request it answers
    board: Board                     # the snapshot that was analysed
    result: Optional[SolveResult]
    recommendation: Optional[Recommendation]
    elapsed_ms: float
    error: Optional[str] = None


class AnalysisWorker(LatestOnlyWorker):
    """Solves and ranks moves for board snapshots, sharing one SolverCache."""

    def __init__(self, parent=None, solve_options: Optional[dict] = None):
        super().__init__(parent)
        self.cache = SolverCache()
        self.solve_options = dict(solve_options or {})

    def handle(self, job: tuple[int, Board]) -> Analysis:
        ticket, board = job
        started = time.perf_counter()
        result = solve(board, cache=self.cache, **self.solve_options)
        rec = recommend(board, result)
        return Analysis(ticket, board, result, rec, (time.perf_counter() - started) * 1000)

    def error_result(self, job, exc: Exception) -> Analysis:
        ticket, board = job
        return Analysis(ticket, board, None, None, 0.0, error=f"Analysis failed: {exc}")
