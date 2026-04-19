import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 2. Распознавание: кадр OpenCV, оверлей MediaPipe, жест и уверенность из KNN (как cv/realtime_infer.py).
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    property string gestureName: qsTr("—")
    property string executedCommand: qsTr("—")
    property bool executeCommands: true

    function commandForKnnClass(cls) {
        var c = (cls || "").toLowerCase()
        if (c === "zoom_one" || c.indexOf("zoom_one") >= 0)
            return "Open Browser"
        if (c === "zoom" || (c.indexOf("zoom") >= 0 && c.indexOf("zoom_one") < 0))
            return "Open Browser"
        if (c.indexOf("hello") >= 0 || c.indexOf("swipe") >= 0)
            return "Open Browser"
        if (c.indexOf("thumb") >= 0 || c.indexOf("up") >= 0)
            return "Volume Up"
        if (c.indexOf("palm") >= 0 || c.indexOf("stop") >= 0)
            return "Close Window"
        return ""
    }

    // Подключение CV при показе экрана в StackView
    StackView.onStatusChanged: {
        if (StackView.status === StackView.Active)
            appController.startEmbeddedGestureRecognition()
        else if (StackView.status === StackView.Deactivating || StackView.status === StackView.Inactive)
            appController.stopEmbeddedGestureRecognition()
    }

    Component.onCompleted: {
        if (StackView.view && StackView.status === StackView.Active)
            appController.startEmbeddedGestureRecognition()
    }

    Connections {
        target: appController

        function onGestureDetected(gesture) {
            gestureName = gesture
            var cmd = commandForKnnClass(gesture)
            if (cmd.length && executeCommands) {
                appController.executeCommand(cmd)
            }
        }

        function onCommandExecuted(command) {
            executedCommand = command
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            Text {
                text: qsTr("Распознавание жестов (2)")
                font.pixelSize: 18
                font.bold: true
                color: Material.foreground
                Layout.fillWidth: true
            }
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 320
            Layout.preferredHeight: 420
            clip: true

            CameraPreview {
                anchors.fill: parent
                previewHeight: parent.height
                cornerRadius: 16
                showStatusBanner: false
                showHandOverlay: true
                handLandmarksJson: appController.embeddedLandmarksJson
            }

            Label {
                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.margins: 6
                text: appController.isCameraActive
                      ? qsTr("Кадр: OpenCV · скелет: MediaPipe · класс: KNN (models/knn.pkl)")
                      : qsTr("Запускается камера…")
                font.pixelSize: 10
                color: "#aaa"
            }
        }

        Label {
            Layout.fillWidth: true
            text: qsTr("Распознанный жест: ") + gestureName
            font.pixelSize: 16
            font.bold: true
            color: Material.foreground
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Label {
                text: qsTr("Уверенность:")
                color: Material.color(Material.Grey, Material.Shade400)
            }

            ProgressBar {
                Layout.fillWidth: true
                from: 0
                to: 1
                value: appController.embeddedRecognitionConfidence
            }

            Label {
                text: Math.round(appController.embeddedRecognitionConfidence * 100) + "%"
                font.bold: true
                color: Material.accent
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Label {
                text: qsTr("Выполнять команды:")
                color: Material.color(Material.Grey, Material.Shade400)
            }

            Switch {
                checked: executeCommands
                onToggled: executeCommands = checked
            }

            Item { Layout.fillWidth: true }

            Label {
                text: qsTr("Две руки:")
                color: Material.color(Material.Grey, Material.Shade400)
            }

            Switch {
                id: twoHandsSwitch
                checked: appController.twoHandsMode
                onToggled: appController.setTwoHandsMode(checked)
                ToolTip.visible: hovered
                ToolTip.delay: 600
                ToolTip.text: qsTr("Распознавать обе руки одновременно (MediaPipe num_hands=2)")
            }

            Connections {
                target: appController
                function onTwoHandsModeChanged() {
                    if (twoHandsSwitch.checked !== appController.twoHandsMode)
                        twoHandsSwitch.checked = appController.twoHandsMode
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 48
            radius: 10
            color: "#2a2a3e"

            Label {
                anchors.fill: parent
                anchors.margins: 10
                text: qsTr("Выполненная команда: ") + executedCommand
                wrapMode: Text.WordWrap
                color: Material.foreground
            }
        }

        Text {
            text: qsTr("Экран использует тот же конвейер, что cv/realtime_infer.py (MediaPipe Hands + окно кадров + KNN), внутри приложения без отдельного окна.")
            font.pixelSize: 11
            color: Material.color(Material.Grey, Material.Shade500)
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Item {
            Layout.fillHeight: true
        }
    }
}
