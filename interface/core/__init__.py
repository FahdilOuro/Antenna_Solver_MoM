"""Front-end logic that does not depend on Qt.

Rule: nothing in this package may import PySide6, pyvista or any GUI library,
so that it can be used from a notebook or a test without opening a window.
"""

from interface.core.mesh_model import SurfaceMesh, MeshImportError, load_msh

__all__ = ["SurfaceMesh", "MeshImportError", "load_msh"]
