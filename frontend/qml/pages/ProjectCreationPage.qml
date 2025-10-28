import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../common"

Item {
    id: projectCreationPage
    
    // Signals
    signal backToHome()
    signal projectCreated(var projectData)
    
    // Properties for form data
    property string antennaName: ""
    property string frequencyType: "single" // "single" or "band"
    property real singleFrequency: 2.4
    property string singleFrequencyUnit: "GHz"
    property real startFrequency: 2.0
    property string startFrequencyUnit: "GHz"
    property real endFrequency: 3.0
    property string endFrequencyUnit: "GHz"
    property string outputPath: ""
    
    ScrollView {
        anchors.fill: parent
        anchors.margins: 20
        
        ColumnLayout {
            width: Math.min(parent.width, 800)
            spacing: 20
            
            // Header with back button
            RowLayout {
                Layout.fillWidth: true
                spacing: 15
                
                Button {
                    text: "← Back"
                    onClicked: projectCreationPage.backToHome()
                }
                
                Label {
                    text: "Create New Project"
                    font.pixelSize: 24
                    font.bold: true
                    color: "#2c3e50"
                }
            }
            
            // Form section
            GroupBox {
                title: "Project Information"
                Layout.fillWidth: true
                
                ColumnLayout {
                    anchors.fill: parent
                    spacing: 15
                    
                    // Antenna name
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 5
                        
                        Label {
                            text: "Antenna Name *"
                            font.bold: true
                        }
                        
                        TextField {
                            id: antennaNameField
                            Layout.fillWidth: true
                            placeholderText: "Enter antenna name (e.g., Dipole_5GHz)"
                            onTextChanged: projectCreationPage.antennaName = text
                        }
                    }
                    
                    // Frequency configuration
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 5
                        
                        Label {
                            text: "Frequency Configuration *"
                            font.bold: true
                        }
                        
                        // Frequency type selection
                        RowLayout {
                            spacing: 20
                            
                            RadioButton {
                                id: singleFreqRadio
                                text: "Single Frequency"
                                checked: true
                                onCheckedChanged: {
                                    if (checked) projectCreationPage.frequencyType = "single"
                                }
                            }
                            
                            RadioButton {
                                id: bandFreqRadio
                                text: "Frequency Band"
                                onCheckedChanged: {
                                    if (checked) projectCreationPage.frequencyType = "band"
                                }
                            }
                        }
                        
                        // Single frequency input
                        RowLayout {
                            visible: singleFreqRadio.checked
                            Layout.fillWidth: true
                            spacing: 10
                            
                            Label {
                                text: "Frequency:"
                            }
                            TextField {
                                id: singleFreqLineEdit
                                text: singleFrequency.toFixed(2)
                                inputMethodHints: Qt.ImhFormattedNumbersOnly

                                // --- User starts typing
                                onTextChanged: {
                                    // Allow only digits and dot while typing
                                    if (!/^[0-9]*\.?[0-9]*$/.test(text)) {
                                        // Remove invalid characters
                                        text = text.replace(/[^0-9.]/g, "")
                                    }
                                    // console.debug("User is typing:", text)
                                }

                                // --- Validation on completion
                                onEditingFinished: {
                                    let newValue = Number(text)
                                    if (!isNaN(newValue) && newValue >= 1.0 && newValue <= 1000.0) {
                                        projectCreationPage.singleFrequency = newValue
                                        // console.log("✅ Frequency validated:", newValue)
                                        text = newValue.toFixed(2) // format nicely
                                    } else {
                                        // console.warn("❌ Invalid input:", text)
                                        // Reset to previous valid value
                                        text = projectCreationPage.singleFrequency.toFixed(2)
                                    }
                                }
                            }
                            ComboBox {
                                id: singleFreqUnitCombo
                                model: ["Hz", "kHz", "MHz", "GHz", "THz"]
                                currentIndex: 2 // Default to GHz
                                
                                onCurrentTextChanged: {
                                    // console.log("Frequency unit changed to:", currentText)
                                    projectCreationPage.singleFrequencyUnit = currentText
                                }
                                
                                Component.onCompleted: {
                                    // console.log("ComboBox initialized with:", currentText)
                                    projectCreationPage.singleFrequencyUnit = currentText
                                }
                            }
                        }
                        
                        // Frequency band inputs
                        ColumnLayout {
                            visible: bandFreqRadio.checked
                            Layout.fillWidth: true
                            spacing: 10
                            
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                
                                Label {
                                    text: "Start:"
                                }
                                
                                TextField {
                                    id: startFreqTextField

                                    // Initial value formatted with 2 decimals
                                    text: (200 / 100).toFixed(2)

                                    inputMethodHints: Qt.ImhFormattedNumbersOnly // numeric keyboard with decimal point

                                    property int decimals: 2
                                    property real realValue: Number(text) || 0

                                    // --- Live input: allow only digits and one dot
                                    onTextChanged: {
                                        if (!/^[0-9]*\.?[0-9]*$/.test(text)) {
                                            text = text.replace(/[^0-9.]/g, "")
                                        }
                                        // console.debug("User is typing:", text)
                                    }

                                    // --- Validation on finish (Enter pressed or focus lost)
                                    onEditingFinished: {
                                        let newValue = Number(text)
                                        let minValue = Math.min(1, 100000) / 100
                                        let maxValue = Math.max(1, 100000) / 100

                                        if (!isNaN(newValue) && newValue >= minValue && newValue <= maxValue) {
                                            projectCreationPage.startFrequency = newValue
                                            // console.log("✅ Start frequency validated:", newValue)
                                            // Format with 2 decimals
                                            text = newValue.toFixed(decimals)
                                        } else {
                                            // console.warn("❌ Invalid input:", text)
                                            // Reset to previous valid value
                                            text = projectCreationPage.startFrequency.toFixed(decimals)
                                        }
                                    }

                                    // Optional: focus feedback
                                    onActiveFocusChanged: {
                                        if (activeFocus) {
                                            console.debug("User started editing start frequency.")
                                        } else {
                                            console.debug("User left start frequency field.")
                                        }
                                    }
                                }
                                ComboBox {
                                    id: startFreqUnitCombo
                                    model: ["Hz", "kHz", "MHz", "GHz", "THz"]
                                    currentIndex: 3 // Default to GHz
                                    
                                    onCurrentTextChanged: {
                                        projectCreationPage.startFrequencyUnit = currentText
                                    }
                                    
                                    Component.onCompleted: {
                                        projectCreationPage.startFrequencyUnit = currentText
                                    }
                                }
                            }
                            
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                
                                Label {
                                    text: "End:"
                                }
                                TextField {
                                    id: endFreqTextField

                                    // Initial value formatted with 2 decimals
                                    text: (300 / 100).toFixed(2)

                                    inputMethodHints: Qt.ImhFormattedNumbersOnly // numeric keyboard with decimal point

                                    property int decimals: 2
                                    property real realValue: Number(text) || 0

                                    // --- Live input: allow only digits and one dot
                                    onTextChanged: {
                                        if (!/^[0-9]*\.?[0-9]*$/.test(text)) {
                                            text = text.replace(/[^0-9.]/g, "")
                                        }
                                        // console.debug("User is typing:", text)
                                    }

                                    // --- Validation on finish (Enter pressed or focus lost)
                                    onEditingFinished: {
                                        let newValue = Number(text)
                                        let minValue = Math.min(1, 100000) / 100
                                        let maxValue = Math.max(1, 100000) / 100

                                        if (!isNaN(newValue) && newValue >= minValue && newValue <= maxValue) {
                                            projectCreationPage.endFrequency = newValue
                                            // console.log("✅ End frequency validated:", newValue)
                                            // Format with 2 decimals
                                            text = newValue.toFixed(decimals)
                                        } else {
                                            // console.warn("❌ Invalid input:", text)
                                            // Reset to previous valid value
                                            text = projectCreationPage.endFrequency.toFixed(decimals)
                                        }
                                    }

                                    // Optional: focus feedback
                                    onActiveFocusChanged: {
                                        if (activeFocus) {
                                            console.debug("User started editing end frequency.")
                                        } else {
                                            console.debug("User left end frequency field.")
                                        }
                                    }
                                }
                                ComboBox {
                                    id: endFreqUnitCombo
                                    model: ["Hz", "kHz", "MHz", "GHz", "THz"]
                                    currentIndex: 3 // Default to GHz
                                    
                                    onCurrentTextChanged: {
                                        projectCreationPage.endFrequencyUnit = currentText
                                    }
                                    
                                    Component.onCompleted: {
                                        projectCreationPage.endFrequencyUnit = currentText
                                    }
                                }
                            }
                        }
                    }
                    
                    // Output path
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 5
                        
                        Label {
                            text: "Output Directory *"
                            font.bold: true
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10
                            
                            TextField {
                                id: outputPathField
                                Layout.fillWidth: true
                                placeholderText: "Select output directory..."
                                readOnly: true
                                text: projectCreationPage.outputPath
                            }
                            
                            Button {
                                text: "Browse..."
                                onClicked: fileDialog.open()
                            }
                        }
                    }
                }
            }
            
            // Action buttons
            RowLayout {
                Layout.fillWidth: true
                Layout.topMargin: 20
                spacing: 10
                
                Item { Layout.fillWidth: true }
                
                Button {
                    text: "Cancel"
                    onClicked: projectCreationPage.backToHome()
                }
                
                Button {
                    text: "Create Project"
                    highlighted: true
                    enabled: projectCreationPage.antennaName.length > 0 && 
                             projectCreationPage.outputPath.length > 0
                    
                    onClicked: {
                        // Validate frequency band
                        if (projectCreationPage.frequencyType === "band" && 
                            projectCreationPage.startFrequency >= projectCreationPage.endFrequency) {
                            notificationPopup.showError("Invalid frequency band: start frequency must be less than end frequency")
                            return
                        }
                        
                        // Prepare project data
                        var projectData = {
                            "antennaName": projectCreationPage.antennaName,
                            "frequencyType": projectCreationPage.frequencyType,
                            "outputPath": projectCreationPage.outputPath
                        }
                        
                        if (projectCreationPage.frequencyType === "single") {
                            projectData["frequency"] = projectCreationPage.singleFrequency
                            projectData["frequencyUnit"] = projectCreationPage.singleFrequencyUnit
                        } else {
                            projectData["startFrequency"] = projectCreationPage.startFrequency
                            projectData["startFrequencyUnit"] = projectCreationPage.startFrequencyUnit
                            projectData["endFrequency"] = projectCreationPage.endFrequency
                            projectData["endFrequencyUnit"] = projectCreationPage.endFrequencyUnit
                        }
                        
                        // Call ProjectManager to create project
                        var success = projectManager.createProject(projectData)
                        
                        if (success) {
                            notificationPopup.showSuccess("Project created successfully!")
                            projectCreationPage.projectCreated(projectData)
                            
                            // Reset form
                            antennaNameField.text = ""
                            outputPathField.text = ""
                            singleFreqRadio.checked = true
                        } else {
                            notificationPopup.showError("Failed to create project. Please check the parameters.")
                        }
                    }
                }
            }
        }
    }
    
    // File dialog for selecting output directory
    FolderDialog {
        id: fileDialog
        title: "Select Output Directory"
        // Don't set currentFolder to avoid Windows path issues
        
        onAccepted: {
            // Remove file:// prefix and handle Windows paths correctly
            var path = selectedFolder.toString()
            if (Qt.platform.os === "windows") {
                path = path.replace(/^file:\/\/\//, "")
            } else {
                path = path.replace(/^file:\/\//, "")
            }
            projectCreationPage.outputPath = path
        }
    }
    
    // Notification popup
    NotificationPopup {
        id: notificationPopup
    }
}