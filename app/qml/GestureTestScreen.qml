import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 12. Тестирование: тот же встроенный конвейер, что на экране 2 (камера + MediaPipe + KNN),
 * плюс журнал сессии, порог уверенности и опциональное выполнение демо-команд.
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    readonly property real confidenceWarnBelow: 0.48

    property string lastGesture: qsTr("—")
    property string mappedCommand: qsTr("—")
    property int detectionCount: 0

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

    function appendSessionRow(gesture, confPct, cmdResolved) {
        var timeStr = Qt.formatTime(new Date(), "HH:mm:ss")
        sessionLog.insert(0, {
                                "t": timeStr,
                                "gesture": gesture,
                                "conf": confPct,
                                "cmd": cmdResolved.length ? cmdResolved : qsTr("—")
                            })
        while (sessionLog.count > 14)
            sessionLog.remove(sessionLog.count - 1)
    }

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
            lastGesture = gesture
            var cmd = commandForKnnClass(gesture)
            mappedCommand = cmd.length ? cmd : qsTr("— (нет демо-привязки)")
            var confPct = Math.round(appController.embeddedRecognitionConfidence * 100)
            detectionCount += 1
            appendSessionRow(gesture, confPct, mappedCommand)
            if (executeCommandsSwitch.checked && cmd.length) {
                appController.executeCommand(cmd)
            }
        }

        function onCommandExecuted(command) {
            lastExecutedCommand = command
        }

        function onEmbeddedLandmarksJsonChanged() {
            handOverlay.requestPaint()
        }
    }

    ListModel {
        id: sessionLog
    }

    property string lastExecutedCommand: qsTr("—")

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            Label {
                text: qsTr("Тестирование (12)")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
                color: Material.foreground
            }
        }

        Label {
            text: qsTr("Живой прогон KNN по кадру с камеры. Журнал фиксирует каждую смену класса; при необходимости включите выполнение демо-команд (осторожно на рабочей машине).")
            font.pixelSize: 12
            color: Material.color(Material.Grey, Material.Shade500)
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: qsTr("Выполнять команды")
                color: Material.color(Material.Grey, Material.Shade400)
            }

            Switch {
                id: executeCommandsSwitch
            }

            Item {
                Layout.fillWidth: true
            }

            Button {
                text: qsTr("Сбросить журнал")
                flat: true
                onClicked: {
                    sessionLog.clear()
                    detectionCount = 0
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 320
            radius: 12
            color: "#151520"
            clip: true

            Item {
                anchors.fill: parent
                anchors.margins: 6

                CameraPreview {
                    anchors.fill: parent
                    previewHeight: 320
                    cornerRadius: 10
                }

                Canvas {
                    id: handOverlay
                    anchors.fill: parent
                    opacity: 0.92
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
                        ctx.strokeStyle = Qt.rgba(0.95, 0.55, 0.15, 0.95)
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
                        ctx.fillStyle = "#FFB74D"
                        for (let i = 0; i < pts.length; i++) {
                            ctx.beginPath()
                            ctx.arc(pts[i][0] * w, pts[i][1] * h, 3.5, 0, 6.28)
                            ctx.fill()
                        }
                    }
                    Component.onCompleted: requestPaint()
                    onWidthChanged: requestPaint()
                    onHeightChanged: requestPaint()
                }

                Rectangle {
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.margins: 8
                    radius: 8
                    color: "#CC000000"
                    width: camBadgeRow.width + 16
                    height: camBadgeRow.height + 10
                    visible: appController.isCameraActive

                    RowLayout {
                        id: camBadgeRow
                        anchors.centerIn: parent
                        spacing: 6

                        Label {
                            text: qsTr("Тест")
                            font.pixelSize: 11
                            font.bold: true
                            color: "#FFB74D"
                        }
                        Label {
                            text: "·"
                            font.pixelSize: 11
                            color: "#aaa"
                        }
                        Label {
                            text: qsTr("событий: ") + detectionCount
                            font.pixelSize: 11
                            color: "#ddd"
                        }
                    }
                }

                Label {
                    anchors.bottom: parent.bottom
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.margins: 6
                    text: appController.isCameraActive
                          ? qsTr("Кадр: OpenCV · скелет: MediaPipe · класс: KNN")
                          : qsTr("Запустите камеру на главном экране (1) или дождитесь автозапуска…")
                    font.pixelSize: 10
                    color: "#aaa"
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Label {
                text: qsTr("Уверенность")
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
                color: appController.embeddedRecognitionConfidence < root.confidenceWarnBelow
                       ? Material.color(Material.Orange, Material.Shade400)
                       : Material.accent
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: lowConfVisible ? 58 : 0
            visible: lowConfVisible
            clip: true
            radius: 8
            color: Material.color(Material.Orange, Material.Shade900)
            border.width: 1
            border.color: Material.color(Material.Orange, Material.Shade600)

            readonly property bool lowConfVisible: appController.isCameraActive && detectionCount > 0
                                                 && appController.embeddedRecognitionConfidence > 0.02
                                                 && appController.embeddedRecognitionConfidence < root.confidenceWarnBelow

            ColumnLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.margins: 8
                spacing: 2

                Label {
                    text: qsTr("Низкая уверенность распознавания")
                    font.bold: true
                    font.pixelSize: 13
                    color: Material.color(Material.Orange, Material.Shade200)
                    Layout.fillWidth: true
                }
                Label {
                    text: qsTr("Добавьте примеры жеста, переобучите модель или улучшите освещение.")
                    font.pixelSize: 11
                    wrapMode: Text.WordWrap
                    color: Material.color(Material.Orange, Material.Shade400)
                    Layout.fillWidth: true
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4

                Label {
                    text: qsTr("Класс KNN")
                    font.pixelSize: 11
                    color: Material.color(Material.Grey, Material.Shade500)
                }
                Label {
                    text: lastGesture
                    font.pixelSize: 16
                    font.bold: true
                    color: Material.foreground
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4

                Label {
                    text: qsTr("Демо-команда")
                    font.pixelSize: 11
                    color: Material.color(Material.Grey, Material.Shade500)
                }
                Label {
                    text: mappedCommand
                    font.pixelSize: 15
                    font.bold: true
                    color: Material.accent
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 46
            radius: 10
            color: "#2a2a3e"

            Label {
                anchors.fill: parent
                anchors.margins: 10
                text: qsTr("Последняя выполненная команда: ") + lastExecutedCommand
                wrapMode: Text.WordWrap
                color: Material.foreground
                font.pixelSize: 13
            }
        }

        Label {
            text: qsTr("Журнал сессии (смена класса)")
            font.bold: true
            color: Material.foreground
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(160, 48 + sessionLog.count * 36)
            radius: 10
            color: "#222230"
            border.width: 1
            border.color: Material.color(Material.Grey, Material.Shade800)
            clip: true

            ListView {
                anchors.fill: parent
                anchors.margins: 6
                model: sessionLog
                spacing: 4
                clip: true

                delegate: Rectangle {
                    width: ListView.view.width
                    height: rowL.height + 10
                    radius: 6
                    color: index % 2 === 0 ? "#2a2a38" : "#252532"

                    RowLayout {
                        id: rowL
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.margins: 6
                        spacing: 8

                        Label {
                            text: model.t
                            font.pixelSize: 11
                            color: Material.color(Material.Grey, Material.Shade500)
                            Layout.preferredWidth: 56
                        }
                        Label {
                            text: model.gesture
                            font.pixelSize: 12
                            font.bold: true
                            color: Material.foreground
                            Layout.preferredWidth: 88
                            elide: Text.ElideRight
                        }
                        Label {
                            text: model.conf + "%"
                            font.pixelSize: 11
                            color: Material.accent
                            Layout.preferredWidth: 36
                        }
                        Label {
                            text: model.cmd
                            font.pixelSize: 11
                            color: Material.color(Material.Grey, Material.Shade300)
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                    }
                }

                Label {
                    anchors.centerIn: parent
                    visible: sessionLog.count === 0
                    text: qsTr("Пока нет событий — покажите жест в камеру.")
                    font.pixelSize: 12
                    color: Material.color(Material.Grey, Material.Shade600)
                }
            }
        }

        Label {
            text: qsTr("Постоянные привязки задаются в экране 6; здесь для питча используется упрощённая таблица имён классов KNN → демо-команды.")
            font.pixelSize: 10
            color: Material.color(Material.Grey, Material.Shade600)
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }

        Item {
            Layout.fillHeight: true
        }
    }
}
