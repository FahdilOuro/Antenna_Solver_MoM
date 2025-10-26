import sys
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QUrl

# Import controllers
from frontend.controllers.project_manager import ProjectManager


def main():
    """
    Main entry point for the Method of Moments solver application.
    Initializes Qt application, registers QML types, and loads the main QML window.
    """
    # Create Qt application instance
    app = QGuiApplication(sys.argv)
    app.setOrganizationName("MoMSolver")
    app.setApplicationName("Method of Moments Solver")
    
    # Create QML engine
    engine = QQmlApplicationEngine()
    
    # Create and register ProjectManager controller
    project_manager = ProjectManager()
    engine.rootContext().setContextProperty("projectManager", project_manager)
    
    # Get the path to the main QML file
    qml_file = Path(__file__).resolve().parent / "frontend" / "qml" / "MainWindow.qml"
    
    # Load the main QML file
    engine.load(QUrl.fromLocalFile(str(qml_file)))
    
    # Check if QML loaded successfully
    if not engine.rootObjects():
        print("Error: Failed to load QML file")
        sys.exit(-1)
    
    # Execute the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()