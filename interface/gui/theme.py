"""Color theme definitions for the GUI of the application.

This module defines the color themes used in the GUI. 
"""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

#-----------------------------------------------------------------------
# 1- Dictionary of color themes
#
#-----------------------------------------------------------------------
LIGHT_THEME = {
    # Qt standard palette colors
    "Window":            "#f5f5f5",
    "WindowText":        "#1e1e1e",
    "Base":              "#ffffff",
    "AlternateBase":     "#f0f0f0",
    "Text":              "#1e1e1e",
    "Button":            "#e0e0e0",
    "ButtonText":        "#1e1e1e",
    "Highlight":         "#308cc6",
    "HighlightedText":   "#ffffff",
    "ToolTipBase":       "#ffffdc",
    "ToolTipText":       "#1e1e1e",
    "PlaceholderText":   "#808080",
    # Disabled colors
    "DisabledText":      "#a0a0a0",
    # 3D View colors
    "View3DBackground":  "#ffffff",
    "View3DEdges":       "#303030",
}

DARK_THEME = {
    "Window":            "#2b2b2b",
    "WindowText":        "#e0e0e0",
    "Base":              "#1e1e1e",
    "AlternateBase":     "#272727",
    "Text":              "#e0e0e0",
    "Button":            "#3c3c3c",
    "ButtonText":        "#e0e0e0",
    "Highlight":         "#4a90d9",
    "HighlightedText":   "#ffffff",
    "ToolTipBase":       "#3c3c3c",
    "ToolTipText":       "#e0e0e0",
    "PlaceholderText":   "#808080",
    # Disabled colors
    "DisabledText":      "#6b6b6b",
    # 3D View colors
    "View3DBackground":  "#1a1a1a",
    "View3DEdges":       "#c0c0c0",
}

THEMES = {
    "Light": LIGHT_THEME,
    "Dark": DARK_THEME,
}

#------------------------------------------------------------------------
# 2) Key mapping from dict to QPalette.ColorRole
#
# Only keys that are real QPalette.ColorRole values go here. "DisabledText"
# is a colour of the Disabled *group* (set by hand in build_palette), and the
# "View3D*" colours are for the PyVista view, returned by apply_theme.
#------------------------------------------------------------------------
_ROLE_MAP = {
    "Window":          QPalette.ColorRole.Window,
    "WindowText":      QPalette.ColorRole.WindowText,
    "Base":            QPalette.ColorRole.Base,
    "AlternateBase":   QPalette.ColorRole.AlternateBase,
    "Text":            QPalette.ColorRole.Text,
    "Button":          QPalette.ColorRole.Button,
    "ButtonText":      QPalette.ColorRole.ButtonText,
    "Highlight":       QPalette.ColorRole.Highlight,
    "HighlightedText": QPalette.ColorRole.HighlightedText,
    "ToolTipBase":     QPalette.ColorRole.ToolTipBase,
    "ToolTipText":     QPalette.ColorRole.ToolTipText,
    "PlaceholderText": QPalette.ColorRole.PlaceholderText,
}

def build_palette(colors: dict) -> QPalette:
    """Build a QPalette from a dictionary of colors.

    Args:
        colors (dict): A dictionary mapping color roles to color values.

    Returns:
        QPalette: The constructed QPalette.
    """
    palette = QPalette()

    # Groupes Active + Inactive : same colors
    for key, role in _ROLE_MAP.items():
        color = QColor(colors[key])
        palette.setColor(QPalette.ColorGroup.Active, role, color)
        palette.setColor(QPalette.ColorGroup.Inactive, role, color)

    # Disabled colors
    disabled_text = QColor(colors["DisabledText"])
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.WindowText,      disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Text,            disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.ButtonText,      disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.HighlightedText, disabled_text)

    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Window, QColor(colors["Window"]))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Base,   QColor(colors["Base"]))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Button, QColor(colors["Button"]))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Highlight, QColor(colors["Button"]))

    return palette

#------------------------------------------------------------------------
# 3) Function to apply a theme to the application
#------------------------------------------------------------------------
def apply_theme(app: QApplication, name: str) -> dict:
    """Apply a color theme to the application.

    Args:
        app (QApplication): The application instance.
        name (str): The name of the theme to apply. Must be one of the keys in THEMES.

    Returns:
        dict: The colours of the 3-D view, keys "background" and "edges".

    Raises:
        ValueError: If the specified theme name is not found in THEMES.
    """
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}; available: {', '.join(THEMES)}")
    colors = THEMES[name]

    app.setStyle("Fusion")
    app.setPalette(build_palette(colors))

    return {
        "background": colors["View3DBackground"],
        "edges":      colors["View3DEdges"],
    }