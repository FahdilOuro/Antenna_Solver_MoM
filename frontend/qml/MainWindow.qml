import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "pages"

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1200
    height: 800
    title: "Method of Moments Solver"
    
    // Property to manage current page
    property string currentPage: "home"
    
    // Menu bar at the top
    menuBar: MenuBar {
        Menu {
            title: qsTr("&File")
            
            Action {
                text: qsTr("&New Project")
                shortcut: StandardKey.New
                onTriggered: {
                    mainWindow.currentPage = "projectCreation"
                }
            }
            
            MenuSeparator {}
            
            Action {
                text: qsTr("&Exit")
                shortcut: StandardKey.Quit
                onTriggered: Qt.quit()
            }
        }
        
        Menu {
            title: qsTr("&Help")
            
            Action {
                text: qsTr("&About")
                onTriggered: {
                    aboutDialog.open()
                }
            }
        }
    }
    
    // Main content area with page loader
    StackLayout {
        id: pageStack
        anchors.fill: parent
        currentIndex: {
            switch(mainWindow.currentPage) {
                case "home": return 0
                case "projectCreation": return 1
                default: return 0
            }
        }
        
        // Home Page
        HomePage {
            id: homePage
            onCreateProjectClicked: {
                mainWindow.currentPage = "projectCreation"
            }
        }
        
        // Project Creation Page
        ProjectCreationPage {
            id: projectCreationPage
            onBackToHome: {
                mainWindow.currentPage = "home"
            }
            onProjectCreated: function(projectData) {
                // Handle project creation
                console.log("Project created:", JSON.stringify(projectData))
                mainWindow.currentPage = "home"
            }
        }
    }
    
    // About Dialog
    Dialog {
        id: aboutDialog
        title: "About MoM Solver"
        anchors.centerIn: parent
        width: 400
        standardButtons: Dialog.Ok
        
        ColumnLayout {
            spacing: 10
            anchors.fill: parent
            
            Label {
                text: "Method of Moments Solver"
                font.bold: true
                font.pixelSize: 18
            }
            
            Label {
                text: "Version 1.0.0"
                font.pixelSize: 14
            }
            
            Label {
                text: "An electromagnetic solver using the Method of Moments technique."
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }
}