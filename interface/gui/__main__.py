"""Entry point:  python -m interface.gui   (from the repository root)."""

import os
import sys

# pyvistaqt finds its Qt binding through `qtpy`, which picks PyQt5 first if it
# is installed (Spyder or some conda packages bring it). The interface is
# written for PySide6, and two Qt bindings in one process crash, so the binding
# is fixed here, before anything imports qtpy.
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from interface.gui.main_window import MainWindow  # noqa: E402
from interface.gui.theme import apply_theme  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Antenna Solver MoM")
    app.setOrganizationName("LEAT")

    view3d_theme = apply_theme(app, "Light")

    window = MainWindow(view3d_theme=view3d_theme)
    window.show()

    # Optional: a .msh path on the command line is opened at start-up.
    if len(sys.argv) > 1:
        window.import_mesh(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
