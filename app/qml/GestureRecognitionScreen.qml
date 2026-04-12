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
        }

        function onCommandExecuted(command) {
            executedCommand = command
        }

        function onEmbeddedLandmarksJsonChanged() {
            handOverlay.requestPaint()
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
                opacity: 0.9
                onPaint: {
                    const ctx = getContext("2d")
                    ctx.reset()
                    let pts = []
                    try {
                        pts = JSON.parse(appController.embeddedLandmarksJson || "[]")
                    } catch (e) {
                        pts = []
                    }
                    if (!pts.length)
                        return
                    const w = width
                    const h = height
                    const edges = [
                        [0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [0, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8],
                        [0, 9],
                        [9, 10],
                        [10, 11],
                        [11, 12],
                        [0, 13],
                        [13, 14],
                        [14, 15],
                        [15, 16],
                        [0, 17],
                        [17, 18],
                        [18, 19],
                        [19, 20],
                        [5, 9],
                        [9, 13],
                        [13, 17]
                    ]
                    ctx.strokeStyle = Qt.rgba(0, 0.85, 0.95, 0.92)
                    ctx.lineWidth = 2.5
                    ctx.beginPath()
                    for (let e = 0; e < edges.length; e++) {
                        const a = edges[e][0]
                        const b = edges[e][1]
                        if (a < pts.length && b < pts.length) {
                            ctx.moveTo(pts[a][0] * w, pts[a][1] * h)
                            ctx.lineTo(pts[b][0] * w, pts[b][1] * h)
                        }
                    }
                    ctx.stroke()
                    ctx.fillStyle = "#00E5FF"
                    for (let i = 0; i < pts.length; i++) {
                        ctx.beginPath()
                        ctx.arc(pts[i][0] * w, pts[i][1] * h, 4, 0, 6.28)
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
