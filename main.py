# main.py
import sys
import os
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine

from frontend.controllers.ProjectController import ProjectController

def main():
    print("🚀 Starting Antenna MoM Solver...")

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    controller = ProjectController()
    engine.rootContext().setContextProperty("controller", controller)

    qml_path = os.path.join(os.path.dirname(__file__), "frontend", "qml", "MainWindow.qml")
    engine.load(QUrl.fromLocalFile(os.path.abspath(qml_path)))

    if not engine.rootObjects():
        return -1
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
