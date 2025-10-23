// frontend/qml/NewProjectDialog.qml
// Enhanced version: supports frequency units and central frequency

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: dialog
    title: "Create New Project"
    modal: true
    standardButtons: Dialog.Ok | Dialog.Cancel
    visible: false
    width: 450
    height: 400

    property var controller

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 12

        Label {
            text: "Project Information"
            font.bold: true
            font.pointSize: 14
            Layout.alignment: Qt.AlignHCenter
        }

        // Project name
        TextField {
            id: projectName
            placeholderText: "Antenna name"
        }

        // Simulation type
        ComboBox {
            id: modeSelector
            model: ["Single frequency", "Frequency band"]
        }

        // Unit selector
        ComboBox {
            id: unitSelector
            model: ["MHz", "GHz", "THz"]
            currentIndex: 1  // default GHz
        }

        // Single frequency input
        RowLayout {
            visible: modeSelector.currentIndex === 0
            spacing: 8
            Label { text: "Frequency:" }
            TextField {
                id: freqSingle
                placeholderText: "Value"
                Layout.fillWidth: true
                inputMethodHints: Qt.ImhFormattedNumbersOnly
            }
        }

        // Frequency band inputs
        ColumnLayout {
            visible: modeSelector.currentIndex === 1
            spacing: 8

            RowLayout {
                spacing: 8
                Label { text: "Start:" }
                TextField {
                    id: freqStart
                    placeholderText: "Start frequency"
                    Layout.fillWidth: true
                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                }
            }

            RowLayout {
                spacing: 8
                Label { text: "Stop:" }
                TextField {
                    id: freqStop
                    placeholderText: "Stop frequency"
                    Layout.fillWidth: true
                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                }
            }

            RowLayout {
                spacing: 8
                Label { text: "Center:" }
                TextField {
                    id: freqCenter
                    placeholderText: "Central frequency"
                    Layout.fillWidth: true
                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                }
            }
        }
    }

    // When user clicks OK
    onAccepted: {
        const data = {
            name: projectName.text,
            mode: modeSelector.currentText,
            unit: unitSelector.currentText,
            freqSingle: freqSingle.text,
            freqStart: freqStart.text,
            freqStop: freqStop.text,
            freqCenter: freqCenter.text
        }
        console.log("New Project Data:", data)
        controller.create_project(data)
    }
}
