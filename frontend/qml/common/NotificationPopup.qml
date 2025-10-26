import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Popup {
    id: notificationPopup
    
    // Properties
    property string message: ""
    property string notificationType: "info" // "success", "error", "warning", "info"
    
    // Position at top center
    parent: Overlay.overlay
    x: (parent.width - width) / 2
    y: 20
    width: Math.min(400, parent.width - 40)
    height: contentLayout.implicitHeight + 40
    
    modal: false
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    
    // Auto-close timer
    Timer {
        id: autoCloseTimer
        interval: 3000
        running: false
        repeat: false
        onTriggered: notificationPopup.close()
    }
    
    // Background with dynamic color
    background: Rectangle {
        color: {
            switch(notificationPopup.notificationType) {
                case "success": return "#d4edda"
                case "error": return "#f8d7da"
                case "warning": return "#fff3cd"
                case "info": return "#d1ecf1"
                default: return "#e2e3e5"
            }
        }
        border.color: {
            switch(notificationPopup.notificationType) {
                case "success": return "#c3e6cb"
                case "error": return "#f5c6cb"
                case "warning": return "#ffeeba"
                case "info": return "#bee5eb"
                default: return "#d6d8db"
            }
        }
        border.width: 1
        radius: 5
    }
    
    // Content
    RowLayout {
        id: contentLayout
        anchors.fill: parent
        anchors.margins: 15
        spacing: 15
        
        // Icon
        Label {
            text: {
                switch(notificationPopup.notificationType) {
                    case "success": return "✓"
                    case "error": return "✗"
                    case "warning": return "⚠"
                    case "info": return "ℹ"
                    default: return "•"
                }
            }
            font.pixelSize: 24
            font.bold: true
            color: {
                switch(notificationPopup.notificationType) {
                    case "success": return "#155724"
                    case "error": return "#721c24"
                    case "warning": return "#856404"
                    case "info": return "#0c5460"
                    default: return "#383d41"
                }
            }
        }
        
        // Message
        Label {
            Layout.fillWidth: true
            text: notificationPopup.message
            wrapMode: Text.WordWrap
            color: {
                switch(notificationPopup.notificationType) {
                    case "success": return "#155724"
                    case "error": return "#721c24"
                    case "warning": return "#856404"
                    case "info": return "#0c5460"
                    default: return "#383d41"
                }
            }
        }
        
        // Close button
        Button {
            text: "×"
            flat: true
            font.pixelSize: 20
            font.bold: true
            Layout.preferredWidth: 30
            Layout.preferredHeight: 30
            onClicked: notificationPopup.close()
        }
    }
    
    // Public functions
    function showSuccess(msg) {
        message = msg
        notificationType = "success"
        open()
        autoCloseTimer.restart()
    }
    
    function showError(msg) {
        message = msg
        notificationType = "error"
        open()
        autoCloseTimer.restart()
    }
    
    function showWarning(msg) {
        message = msg
        notificationType = "warning"
        open()
        autoCloseTimer.restart()
    }
    
    function showInfo(msg) {
        message = msg
        notificationType = "info"
        open()
        autoCloseTimer.restart()
    }
}