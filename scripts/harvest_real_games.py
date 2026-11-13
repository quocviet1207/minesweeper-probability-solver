"""Records labelled frames from a real web Minesweeper for digit training/testing.

Loads the page in an off-screen browser, plays it with the solver (reading the
true board from the page so play goes deep and many high digits appear),
and for every step saves the screenshot together with the page's own cell
states:

    python scripts/harvest_real_games.py OUT_DIR --seed 1 --zoom 1.4 --steps 60

Only for minesweeperonline.com's markup (cells are elements with id "row_col"
and classes like "square open3"). Frames go to OUT_DIR/<name>_NNN.png with a
matching .json holding the true states (Board cell codes).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWebEngineCore import QWebEnginePage  # noqa: E402
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from minesweeper_solver.core import FLAG, MINE, UNREVEALED, Board, solve  # noqa: E402

URL = "https://minesweeperonline.com/"
STATE_JS = """(function(){var out=[];for(var r=1;r<60;r++){var row=[];for(var c=1;c<80;c++){
var e=document.getElementById(r+'_'+c); if(!e||e.style.display=='none') break; row.push(e.className);}
if(!row.length) break; out.push(row);} return JSON.stringify(out);})()"""


def press(target_id: str, button: int) -> str:
    return ("(function(){var e=document.getElementById('%s'); if(!e) return 'missing';"
            "['mousedown','mouseup'].forEach(function(t){e.dispatchEvent(new MouseEvent(t,"
            "{bubbles:true,cancelable:true,view:window,button:%d}));}); return 'ok';})()" % (target_id, button))


def decode(name: str):
    name = name.replace("square", "").strip()
    if name == "blank":
        return UNREVEALED
    if name.startswith("open"):
        return int(name[4:])
    if name == "bombflagged":
        return FLAG
    if name in ("bombrevealed", "bombdeath"):
        return MINE
    return None                       # e.g. a wrong flag shown after a loss


class QuietPage(QWebEnginePage):
    def javaScriptAlert(self, origin, msg):
        pass

    def javaScriptConfirm(self, origin, msg):
        return False

    def javaScriptPrompt(self, origin, msg, default):
        return (False, "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"s{args.seed}_z{int(args.zoom * 100)}"
    rng = random.Random(args.seed)

    app = QApplication([])
    view = QWebEngineView()
    view.setPage(QuietPage(view))
    view.resize(1150, 820)
    view.setZoomFactor(args.zoom)
    view.show()
    view.load(QUrl(URL))
    state = {"step": 0, "saved": 0}

    def grab() -> QImage:
        return view.grab().toImage().convertToFormat(QImage.Format_RGB888)

    def on_state(raw):
        if not raw:
            QTimer.singleShot(500, read_state)
            return
        names = json.loads(raw)
        truth = [[decode(n) for n in row] for row in names]
        step = state["step"]
        grab().save(os.path.join(args.out, f"{name}_{step:03d}.png"))
        with open(os.path.join(args.out, f"{name}_{step:03d}.json"), "w") as f:
            json.dump({"truth": truth}, f)
        state["saved"] += 1
        state["step"] += 1
        if state["step"] >= args.steps:
            print(f"{name}: saved {state['saved']} frames", flush=True)
            app.quit()
            return
        act(truth)

    def act(truth):
        flat = [v for row in truth for v in row]
        if MINE in flat or UNREVEALED not in flat:        # lost or won: new game
            js = press("face", 0)
        else:
            rows, cols = len(truth), len(truth[0])
            board = Board(rows, cols, 99)
            board.cells = [[UNREVEALED if v is None else v for v in row] for row in truth]
            result = solve(board)
            if result.mine_cells and rng.random() < 0.5:
                r, c = rng.choice(result.mine_cells)
                js = press(f"{r + 1}_{c + 1}", 2)
            elif result.safe_cells:
                r, c = rng.choice(result.safe_cells)
                js = press(f"{r + 1}_{c + 1}", 0)
            else:
                covered = [cell for cell, p in result.probs.items()]
                r, c = min(covered, key=lambda cell: (result.probs[cell], rng.random()))
                js = press(f"{r + 1}_{c + 1}", 0)
        view.page().runJavaScript(js, lambda _r: QTimer.singleShot(350, read_state))

    def read_state():
        view.page().runJavaScript(STATE_JS, on_state)

    QTimer.singleShot(8000, read_state)
    QTimer.singleShot(8000 + args.steps * 1500 + 30000, app.quit)
    app.exec()


if __name__ == "__main__":
    main()
