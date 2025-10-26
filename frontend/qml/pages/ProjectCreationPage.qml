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
    property real startFrequency: 2.0
    property real endFrequency: 3.0
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
                            
                            SpinBox {
                                id: singleFreqSpinBox
                                from: 1
                                to: 100000
                                value: 2400
                                editable: true
                                
                                property int decimals: 2
                                property real realValue: value / 100
                                
                                validator: DoubleValidator {
                                    bottom: Math.min(singleFreqSpinBox.from, singleFreqSpinBox.to)
                                    top: Math.max(singleFreqSpinBox.from, singleFreqSpinBox.to)
                                }
                                
                                textFromValue: function(value, locale) {
                                    return Number(value / 100).toLocaleString(locale, 'f', decimals)
                                }
                                
                                valueFromText: function(text, locale) {
                                    return Number.fromLocaleString(locale, text) * 100
                                }
                                
                                onValueChanged: {
                                    projectCreationPage.singleFrequency = realValue
                                }
                            }
                            
                            Label {
                                text: "GHz"
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
                                
                                SpinBox {
                                    id: startFreqSpinBox
                                    from: 1
                                    to: 100000
                                    value: 200
                                    editable: true
                                    
                                    property int decimals: 2
                                    property real realValue: value / 100
                                    
                                    validator: DoubleValidator {
                                        bottom: Math.min(startFreqSpinBox.from, startFreqSpinBox.to)
                                        top: Math.max(startFreqSpinBox.from, startFreqSpinBox.to)
                                    }
                                    
                                    textFromValue: function(value, locale) {
                                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                                    }
                                    
                                    valueFromText: function(text, locale) {
                                        return Number.fromLocaleString(locale, text) * 100
                                    }
                                    
                                    onValueChanged: {
                                        projectCreationPage.startFrequency = realValue
                                    }
                                }
                                
                                Label {
                                    text: "GHz"
                                }
                            }
                            
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                
                                Label {
                                    text: "End:"
                                }
                                
                                SpinBox {
                                    id: endFreqSpinBox
                                    from: 1
                                    to: 100000
                                    value: 300
                                    editable: true
                                    
                                    property int decimals: 2
                                    property real realValue: value / 100
                                    
                                    validator: DoubleValidator {
                                        bottom: Math.min(endFreqSpinBox.from, endFreqSpinBox.to)
                                        top: Math.max(endFreqSpinBox.from, endFreqSpinBox.to)
                                    }
                                    
                                    textFromValue: function(value, locale) {
                                        return Number(value / 100).toLocaleString(locale, 'f', decimals)
                                    }
                                    
                                    valueFromText: function(text, locale) {
                                        return Number.fromLocaleString(locale, text) * 100
                                    }
                                    
                                    onValueChanged: {
                                        projectCreationPage.endFrequency = realValue
                                    }
                                }
                                
                                Label {
                                    text: "GHz"
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
                        } else {
                            projectData["startFrequency"] = projectCreationPage.startFrequency
                            projectData["endFrequency"] = projectCreationPage.endFrequency
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