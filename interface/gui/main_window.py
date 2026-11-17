"""Main window: model tree, properties and console docked around a 3-D view.

Milestone 1: import a .msh file and display it. No solver call yet.
"""

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDockWidget, QFileDialog, QHeaderView, QMainWindow, QMessageBox,
    QPlainTextEdit, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
)
from pyvistaqt import QtInteractor

from interface.core import MeshImportError, SurfaceMesh, load_msh
from interface.gui.mesh_view import to_polydata


def _mm(value_m: float) -> str:
    """Format a length given in metres as millimetres."""
    return f"{value_m * 1e3:.3g} mm"


class MainWindow(QMainWindow):

    def __init__(self, view3d_theme: dict | None = None):
        """view3d_theme: colours of the 3-D view, as returned by
        interface.gui.theme.apply_theme ("background", "edges")."""
        super().__init__()
        self.setWindowTitle("Antenna Solver MoM")
        self.resize(1400, 850)

        self.mesh: SurfaceMesh | None = None
        self._show_edges = True
        self.view3d_theme = view3d_theme or {"background": "#f3f3f3", "edges": "#1a1a1a"}

        # Central widget: the 3-D view (a VTK render window inside Qt).
        self.plotter = QtInteractor(self)
        self.plotter.set_background(self.view3d_theme["background"])
        self.plotter.add_axes()
        self.setCentralWidget(self.plotter.interactor)

        self._build_docks()
        self._build_menus()
        self.resizeDocks([self.dock_tree], [380], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_console], [150], Qt.Orientation.Vertical)
        self.statusBar().showMessage("No mesh loaded — File > Import mesh…")
        self.log("Ready.")

    # ------------------------------------------------------------------ layout

    def _build_docks(self):
        # Model tree (left, top). Only "Mesh" is active in milestone 1.
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree_items = {}
        for name in ("Geometry", "Mesh", "Ports", "Studies"):
            item = QTreeWidgetItem([name])
            if name != "Mesh":
                item.setDisabled(True)  # placeholders for the next milestones
            self.tree.addTopLevelItem(item)
            self.tree_items[name] = item
        self.dock_tree = self._add_dock("Project", self.tree, Qt.DockWidgetArea.LeftDockWidgetArea)

        # Properties (left, below the tree): read-only two-column table.
        self.properties = QTableWidget(0, 2)
        self.properties.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties.verticalHeader().setVisible(False)
        self.properties.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.properties.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.dock_properties = self._add_dock("Properties", self.properties, Qt.DockWidgetArea.LeftDockWidgetArea)

        # Console (bottom): messages of the application, later the solver log.
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.dock_console = self._add_dock("Console", self.console, Qt.DockWidgetArea.BottomDockWidgetArea)

    def _add_dock(self, title, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)  # needed later to save/restore the layout
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        act_import = QAction("&Import mesh…", self)
        act_import.setShortcut(QKeySequence("Ctrl+I"))
        act_import.triggered.connect(self._on_import_mesh)
        file_menu.addAction(act_import)
        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = self.menuBar().addMenu("&View")
        act_fit = QAction("&Fit to window", self)
        act_fit.setShortcut(QKeySequence("F"))
        act_fit.triggered.connect(self.plotter.reset_camera)
        view_menu.addAction(act_fit)
        act_edges = QAction("Show mesh &edges", self, checkable=True, checked=True)
        act_edges.toggled.connect(self._on_toggle_edges)
        view_menu.addAction(act_edges)
        view_menu.addSeparator()
        for dock in (self.dock_tree, self.dock_properties, self.dock_console):
            view_menu.addAction(dock.toggleViewAction())  # show/hide each panel

    # ----------------------------------------------------------------- actions

    def _on_import_mesh(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import mesh", "", "Gmsh mesh (*.msh);;All files (*)")
        if filename:
            self.import_mesh(filename)

    def import_mesh(self, filename):
        """Read a .msh file and display it. Writes nothing to disk."""
        self.log(f"Importing {filename} …")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mesh = load_msh(filename)
        except MeshImportError as error:
            QApplication.restoreOverrideCursor()
            self.log(f"Error: {error}")
            QMessageBox.critical(self, "Import failed", str(error))
            return
        QApplication.restoreOverrideCursor()

        self.mesh = mesh
        self._display_mesh()
        self._fill_properties()
        self._update_tree()
        self.statusBar().showMessage(
            f"{mesh.source.name} — {mesh.n_triangles} triangles, {mesh.n_points} nodes")
        self.log(f"Loaded {mesh.n_triangles} triangles, {mesh.n_points} nodes, "
                 f"{mesh.n_surfaces} surface(s).")

    def _on_toggle_edges(self, checked):
        self._show_edges = checked
        if self.mesh is not None:
            self._display_mesh(keep_camera=True)

    # ----------------------------------------------------------------- display

    def _display_mesh(self, keep_camera=False):
        camera = self.plotter.camera_position if keep_camera else None
        self.plotter.clear_actors()
        self.plotter.add_mesh(
            to_polydata(self.mesh),
            scalars="colour",           # one colour per Gmsh surface
            rgb=True,
            show_scalar_bar=False,
            show_edges=self._show_edges,
            edge_color=self.view3d_theme["edges"],
            line_width=0.5,
        )
        if camera is not None:
            self.plotter.camera_position = camera
        else:
            self.plotter.reset_camera()

    def _fill_properties(self):
        m = self.mesh
        lo, hi = m.bounding_box()
        size = hi - lo
        shortest, longest = m.edge_length_range()
        rows = [
            ("File", m.source.name),
            ("Nodes", f"{m.n_points}"),
            ("Triangles", f"{m.n_triangles}"),
            ("Surfaces", f"{m.n_surfaces}"),
            ("Shared edges (≈ unknowns)", f"{m.n_shared_edges()}"),
            ("Size x × y × z", f"{_mm(size[0])} × {_mm(size[1])} × {_mm(size[2])}"),
            ("Edge length min / max", f"{_mm(shortest)} / {_mm(longest)}"),
        ]
        self.properties.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.properties.setItem(row, 0, QTableWidgetItem(name))
            self.properties.setItem(row, 1, QTableWidgetItem(value))

    def _update_tree(self):
        item = self.tree_items["Mesh"]
        item.takeChildren()
        item.addChild(QTreeWidgetItem([self.mesh.source.name]))
        item.setExpanded(True)

    def log(self, message: str):
        self.console.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")

    def closeEvent(self, event):
        self.plotter.close()  # release the VTK render window cleanly
        super().closeEvent(event)
