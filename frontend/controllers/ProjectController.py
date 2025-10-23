# frontend/controllers/ProjectController.py
# Handles project creation logic (directory + config file)

import os
import json
from PySide6.QtCore import QObject, Slot

class ProjectController(QObject):
    def __init__(self, backend=None):
        super().__init__()
        self.backend = backend
        self.projects_dir = os.path.join(os.getcwd(), "data", "projects")

        # Create base directory if missing
        os.makedirs(self.projects_dir, exist_ok=True)

    @Slot("QVariantMap")
    def create_project(self, project_data):
        project_name = project_data.get("name") or "Unnamed_Project"
        project_path = os.path.join(self.projects_dir, project_name)

        # Create folder
        os.makedirs(project_path, exist_ok=True)

        # Prepare configuration
        config = {
            "project_name": project_name,
            "mode": project_data.get("mode"),
            "unit": project_data.get("unit"),
            "frequencies": {
                "single": project_data.get("freqSingle"),
                "start": project_data.get("freqStart"),
                "stop": project_data.get("freqStop"),
                "center": project_data.get("freqCenter"),
            },
        }

        # Save to project.json
        config_path = os.path.join(project_path, "project.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        print(f"[+] Project created at: {project_path}")
        print(f"    → Configuration saved to {config_path}")
