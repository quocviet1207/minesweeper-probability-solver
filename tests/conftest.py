import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def app():
    """One QApplication for the whole test session (Qt allows only one)."""
    return QApplication.instance() or QApplication([])
