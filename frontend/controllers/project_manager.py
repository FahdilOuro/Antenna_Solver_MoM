from PySide6.QtCore import QObject, Slot, Signal
from pathlib import Path
import json
from datetime import datetime


class ProjectManager(QObject):
    """
    Controller class to manage MoM solver projects.
    Handles project creation, saving, loading, and validation.
    """
    
    # Signals
    projectCreated = Signal(str)  # Emitted when a project is successfully created
    projectLoadFailed = Signal(str)  # Emitted when project loading fails
    
    def __init__(self, parent=None):
        """Initialize the ProjectManager."""
        super().__init__(parent)
        self.current_project = None
        self.recent_projects = []
        self._load_recent_projects()
    
    @Slot(dict, result=bool)
    def createProject(self, project_data):
        """
        Create a new MoM solver project.
        
        Args:
            project_data (dict): Dictionary containing project information
                - antennaName: Name of the antenna
                - frequencyType: "single" or "band"
                - frequency: Single frequency value (if frequencyType is "single")
                - startFrequency: Start frequency (if frequencyType is "band")
                - endFrequency: End frequency (if frequencyType is "band")
                - outputPath: Path to save results
        
        Returns:
            bool: True if project created successfully, False otherwise
        """
        try:
            # Validate required fields
            if not self._validate_project_data(project_data):
                print("Error: Invalid project data")
                return False
            
            # Create project structure
            project = {
                "metadata": {
                    "name": project_data.get("antennaName"),
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "configuration": {
                    "antenna": {
                        "name": project_data.get("antennaName")
                    },
                    "frequency": self._extract_frequency_config(project_data),
                    "output": {
                        "path": project_data.get("outputPath")
                    }
                },
                "results": None
            }
            
            # Save project to file
            output_path = Path(project_data.get("outputPath"))
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            
            project_file = output_path / f"{project_data.get('antennaName')}_project.json"
            
            with open(project_file, 'w') as f:
                json.dump(project, f, indent=4)
            
            # Update current project
            self.current_project = project
            self.current_project["file_path"] = str(project_file)
            
            # Add to recent projects
            self._add_to_recent_projects(str(project_file))
            
            # Emit success signal
            self.projectCreated.emit(str(project_file))
            
            print(f"Project created successfully: {project_file}")
            return True
            
        except Exception as e:
            print(f"Error creating project: {str(e)}")
            return False
    
    def _validate_project_data(self, project_data):
        """
        Validate project data before creation.
        
        Args:
            project_data (dict): Project data to validate
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check required fields
        required_fields = ["antennaName", "frequencyType", "outputPath"]
        for field in required_fields:
            if field not in project_data or not project_data[field]:
                return False
        
        # Validate frequency configuration
        freq_type = project_data.get("frequencyType")
        if freq_type == "single":
            if "frequency" not in project_data:
                return False
        elif freq_type == "band":
            if "startFrequency" not in project_data or "endFrequency" not in project_data:
                return False
            if project_data["startFrequency"] >= project_data["endFrequency"]:
                return False
        else:
            return False
        
        return True
    
    def _extract_frequency_config(self, project_data):
        """
        Extract frequency configuration from project data.
        
        Args:
            project_data (dict): Project data
        
        Returns:
            dict: Frequency configuration
        """
        freq_config = {
            "type": project_data.get("frequencyType")
        }
        
        if project_data.get("frequencyType") == "single":
            value = project_data.get("frequency")
            unit = project_data.get("frequencyUnit", "MHz")
            
            freq_config["value"] = value
            freq_config["unit"] = unit
            freq_config["real_value"] = self._convert_to_hz(value, unit)
        else:
            start_value = project_data.get("startFrequency")
            start_unit = project_data.get("startFrequencyUnit", "MHz")
            end_value = project_data.get("endFrequency")
            end_unit = project_data.get("endFrequencyUnit", "MHz")
            
            freq_config["start"] = {
                "value": start_value,
                "unit": start_unit,
                "real_value": self._convert_to_hz(start_value, start_unit)
            }
            freq_config["end"] = {
                "value": end_value,
                "unit": end_unit,
                "real_value": self._convert_to_hz(end_value, end_unit)
            }
        
        return freq_config
    
    def _convert_to_hz(self, value, unit):
        """
        Convert frequency value to Hz.
        
        Args:
            value (float): Frequency value
            unit (str): Frequency unit (Hz, kHz, MHz, GHz, THz)
        
        Returns:
            float: Frequency in Hz
        """
        conversions = {
            "Hz": 1,
            "kHz": 1e3,
            "MHz": 1e6,
            "GHz": 1e9,
            "THz": 1e12
        }
        
        return value * conversions.get(unit, 1)
    
    @Slot(str, result=bool)
    def loadProject(self, project_path):
        """
        Load an existing project from file.
        
        Args:
            project_path (str): Path to the project file
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            project_file = Path(project_path)
            if not project_file.exists():
                print(f"Error: Project file not found: {project_path}")
                self.projectLoadFailed.emit("Project file not found")
                return False
            
            with open(project_file, 'r') as f:
                project = json.load(f)
            
            self.current_project = project
            self.current_project["file_path"] = project_path
            
            # Update recent projects
            self._add_to_recent_projects(project_path)
            
            print(f"Project loaded successfully: {project_path}")
            return True
            
        except Exception as e:
            print(f"Error loading project: {str(e)}")
            self.projectLoadFailed.emit(str(e))
            return False
    
    @Slot(result=dict)
    def getCurrentProject(self):
        """
        Get the current project data.
        
        Returns:
            dict: Current project data or None
        """
        return self.current_project
    
    @Slot(result=list)
    def getRecentProjects(self):
        """
        Get list of recent projects.
        
        Returns:
            list: List of recent project paths
        """
        return self.recent_projects
    
    def _add_to_recent_projects(self, project_path):
        """
        Add a project to the recent projects list.
        
        Args:
            project_path (str): Path to the project file
        """
        if project_path in self.recent_projects:
            self.recent_projects.remove(project_path)
        
        self.recent_projects.insert(0, project_path)
        
        # Keep only last 10 projects
        self.recent_projects = self.recent_projects[:10]
        
        self._save_recent_projects()
    
    def _load_recent_projects(self):
        """Load recent projects from config file."""
        try:
            config_dir = Path.home() / ".mom_solver"
            config_file = config_dir / "recent_projects.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.recent_projects = json.load(f)
        except Exception as e:
            print(f"Error loading recent projects: {str(e)}")
            self.recent_projects = []
    
    def _save_recent_projects(self):
        """Save recent projects to config file."""
        try:
            config_dir = Path.home() / ".mom_solver"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = config_dir / "recent_projects.json"
            
            with open(config_file, 'w') as f:
                json.dump(self.recent_projects, f, indent=4)
        except Exception as e:
            print(f"Error saving recent projects: {str(e)}")