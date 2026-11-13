"""A layout that places items left to right and wraps onto new lines when the
width runs out (Qt's classic flow layout example), centring each line."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidgetItem


class FlowLayout(QLayout):
    def __init__(self, parent=None, h_spacing: int = 8, v_spacing: int = 4):
        super().__init__(parent)
        self._items: list = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout interface ---------------------------------------------------
    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:            # as narrow as the widest single item
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    # -- placement -----------------------------------------------------------
    def _arrange(self, rect: QRect, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        lines: list[list] = [[]]
        widths = [0]
        for item in self._items:
            w = item.sizeHint().width()
            needed = w if not lines[-1] else widths[-1] + self._h + w
            if lines[-1] and needed > area.width():
                lines.append([])
                widths.append(0)
                needed = w
            lines[-1].append(item)
            widths[-1] = needed
        y = area.y()
        for line, width in zip(lines, widths):
            if not line:
                continue
            height = max(item.sizeHint().height() for item in line)
            x = area.x() + max(0, (area.width() - width) // 2)
            for item in line:
                hint = item.sizeHint()
                if apply:
                    item.setGeometry(QRect(QPoint(x, y + (height - hint.height()) // 2), hint))
                x += hint.width() + self._h
            y += height + self._v
        return y - self._v - rect.y() + margins.bottom()
