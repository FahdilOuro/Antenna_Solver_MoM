import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: homePage
    
    // Signal to notify when user wants to create a new project
    signal createProjectClicked()
    
    ColumnLayout {
        anchors.centerIn: parent
        spacing: 30
        width: parent.width * 0.6
        
        // Welcome section
        ColumnLayout {
            spacing: 10
            Layout.alignment: Qt.AlignHCenter
            
            Label {
                text: "Method of Moments Solver"
                font.pixelSize: 32
                font.bold: true
                color: "#2c3e50"
                Layout.alignment: Qt.AlignHCenter
            }
            
            Label {
                text: "Electromagnetic Analysis Tool"
                font.pixelSize: 16
                color: "#7f8c8d"
                Layout.alignment: Qt.AlignHCenter
            }
        }
        
        // Spacer
        Item { Layout.preferredHeight: 20 }
        
        // Quick actions section
        ColumnLayout {
            spacing: 15
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            
            Label {
                text: "Quick Actions"
                font.pixelSize: 18
                font.bold: true
                color: "#34495e"
            }
            
            // New Project Button
            Button {
                text: "Create New Project"
                font.pixelSize: 14
                Layout.preferredWidth: 250
                Layout.preferredHeight: 50
                highlighted: true
                
                onClicked: {
                    homePage.createProjectClicked()
                }
            }
            
            // Open Project Button (placeholder for future)
            Button {
                text: "Open Existing Project"
                font.pixelSize: 14
                Layout.preferredWidth: 250
                Layout.preferredHeight: 50
                enabled: false // Will be enabled when functionality is added
                
                onClicked: {
                    // TODO: Implement open project functionality
                    console.log("Open project clicked")
                }
            }
        }
        
        // Spacer
        Item { Layout.preferredHeight: 30 }
        
        // Recent projects section (placeholder)
        ColumnLayout {
            spacing: 10
            Layout.fillWidth: true
            
            Label {
                text: "Recent Projects"
                font.pixelSize: 16
                font.bold: true
                color: "#34495e"
            }
            
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 150
                color: "#ecf0f1"
                radius: 5
                border.color: "#bdc3c7"
                border.width: 1
                
                Label {
                    anchors.centerIn: parent
                    text: "No recent projects"
                    color: "#95a5a6"
                    font.italic: true
                }
            }
        }
    }
}