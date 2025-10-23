// frontend/qml/MainWindow.qml
// Main application window - styled with dark theme and sidebar

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Material

ApplicationWindow {
    id: mainWindow
    title: "Antenna MoM Solver"
    visible: true
    width: 1100
    height: 700

    Material.theme: Material.Dark
    Material.accent: Material.Teal
    color: "#1E1E1E"

    property var controller

    // Layout
    RowLayout {
        anchors.fill: parent

        // Sidebar
        Rectangle {
            id: sidebar
            width: 220
            color: "#252526"
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 15

                Label {
                    text: "Antenna MoM Solver"
                    color: "white"
                    font.bold: true
                    font.pointSize: 14
                    Layout.alignment: Qt.AlignHCenter
                }

                Rectangle { height: 1; color: "#444" }

                Button {
                    text: "🏠  Home"
                    onClicked: stackView.currentIndex = 0
                }

                Button {
                    text: "📂  Projects"
                    onClicked: stackView.currentIndex = 1
                }

                Button {
                    text: "⚙️  Settings"
                    onClicked: stackView.currentIndex = 2
                }

                Item { Layout.fillHeight: true }

                Button {
                    text: "➕  Create New Project"
                    onClicked: newProjectDialog.visible = true
                }
            }
        }

        // Main content
        StackLayout {
            id: stackView
            Layout.fillWidth: true
            Layout.fillHeight: true

            Rectangle {
                color: "#1E1E1E"
                Label {
                    anchors.centerIn: parent
                    text: "Welcome to Antenna MoM Solver"
                    color: "white"
                    font.pointSize: 18
                }
            }

            Rectangle {
                color: "#1E1E1E"
                Label {
                    anchors.centerIn: parent
                    text: "Projects Section"
                    color: "white"
                }
            }

            Rectangle {
                color: "#1E1E1E"
                Label {
                    anchors.centerIn: parent
                    text: "Settings Section"
                    color: "white"
                }
            }
        }
    }

    // Include New Project Dialog
    NewProjectDialog {
        id: newProjectDialog
        controller: mainWindow.controller
    }
}
