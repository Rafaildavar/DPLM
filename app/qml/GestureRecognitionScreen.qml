import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 2. Распознавание: кадр, оверлей ключевых точек, имя жеста, уверенность, команда.
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost

    property string gestureName: qsTr("—")
    property double confidence: 0
    property string executedCommand: qsTr("—")

    Timer {
        interval: 900
        repeat: true
        running: appController.isRecognizing
        onTriggered: {
            confidence = 0.55 + Math.random() * 0.4
            gestureName = [qsTr("Свайп вправо"), qsTr("Большой палец"), qsTr("Кулак")][Math.floor(Math.random() * 3)]
        }
    }

    Connections {
        target: appController

        function onGestureDetected(gesture) {
            gestureName = gesture
            confidence = 0.92
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
            Layout.preferredHeight: 240
            clip: true

            CameraPreview {
                anchors.fill: parent
                previewHeight: 240
                cornerRadius: 16
            }

            Canvas {
                id: handOverlay
                anchors.fill: parent
                opacity: 0.85
                onPaint: {
                    const ctx = getContext("2d")
                    ctx.reset()
                    ctx.strokeStyle = Qt.rgba(0, 0.85, 0.95, 0.9)
                    ctx.lineWidth = 3
                    const w = width
                    const h = height
                    const cx = w * 0.45
                    const cy = h * 0.42
                    const pts = [
                        [cx, cy],
                        [cx - 30, cy + 50],
                        [cx + 25, cy + 45],
                        [cx - 20, cy + 100],
                        [cx + 35, cy + 95],
                        [cx, cy + 30],
                        [cx + 15, cy + 75]
                    ]
                    ctx.beginPath()
                    ctx.moveTo(pts[0][0], pts[0][1])
                    for (let i = 1; i < pts.length; i++)
                        ctx.lineTo(pts[i][0], pts[i][1])
                    ctx.stroke()
                    for (let i = 0; i < pts.length; i++) {
                        ctx.beginPath()
                        ctx.arc(pts[i][0], pts[i][1], 5, 0, 6.28)
                        ctx.fillStyle = "#00E5FF"
                        ctx.fill()
                    }
                }
                Component.onCompleted: requestPaint()
                onWidthChanged: requestPaint()
                onHeightChanged: requestPaint()
            }

            Label {
                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.margins: 6
                text: qsTr("Оверлей руки — демо; кадр — OpenCV при включённой камере")
                font.pixelSize: 10
                color: "#aaa"
                visible: appController.isCameraActive
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
                value: confidence
            }

            Label {
                text: Math.round(confidence * 100) + "%"
                font.bold: true
                color: Material.accent
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
            text: qsTr("Нижняя панель главного окна запускает фоновое распознавание (cv/realtime_infer.py).")
            font.pixelSize: 11
            color: Material.color(Material.Grey, Material.Shade500)
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }

        Item {
            Layout.fillHeight: true
        }
    }
}
