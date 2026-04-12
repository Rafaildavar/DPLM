import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 4. Запись обучающих примеров: видео, счётчик, старт/стоп, подсказки по положению руки.
 */
Item {
    id: root

    required property StackView gestStack
    property string gestureName: ""
    property var navRoot
    property var pitchHost
    property var gestureCatalog

    property bool isRecording: false
    property int samplesRecorded: 0
    readonly property int targetSamples: 20

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: {
                    finalizeRecording()
                    recordingTimer.stop()
                    isRecording = false
                    if (pitchHost && pitchHost.handleBack)
                        pitchHost.handleBack(gestStack)
                    else
                        gestStack.pop()
                }
            }

            Text {
                text: qsTr("Запись примеров (4)")
                font.pixelSize: 18
                font.bold: true
                color: Material.foreground
                Layout.fillWidth: true
            }
        }

        Text {
            text: gestureName.length > 0 ? (qsTr("Жест: ") + gestureName) : qsTr("Жест: —")
            font.pixelSize: 14
            color: Material.accent
        }

        Item {
            Layout.fillWidth: true
            Layout.preferredHeight: 200

            CameraPreview {
                anchors.fill: parent
                previewHeight: 200
                cornerRadius: 16
            }

            Rectangle {
                anchors.fill: parent
                radius: 16
                color: "transparent"
                border.width: isRecording ? 3 : 0
                border.color: "#E91E63"
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 72
            radius: 12
            color: "#243040"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 6

                Text {
                    text: qsTr("Подсказка")
                    font.bold: true
                    font.pixelSize: 12
                    color: Material.accent
                }

                Text {
                    text: qsTr("Держите руку целиком в кадре, ладонью к камере. Между образцами слегка меняйте угол.")
                    wrapMode: Text.WordWrap
                    font.pixelSize: 13
                    color: Material.foreground
                    Layout.fillWidth: true
                }
            }
        }

        Label {
            text: qsTr("Записано образцов: ") + samplesRecorded + " / " + targetSamples
            font.pixelSize: 16
            font.bold: true
            color: Material.foreground
        }

        ProgressBar {
            Layout.fillWidth: true
            from: 0
            to: targetSamples
            value: samplesRecorded
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Button {
                Layout.fillWidth: true
                text: qsTr("Начать запись")
                enabled: !isRecording
                onClicked: {
                    isRecording = true
                    recordingTimer.start()
                    appController.startGestureTraining(gestureName)
                }
            }

            Button {
                Layout.fillWidth: true
                text: qsTr("Остановить запись")
                enabled: isRecording
                onClicked: {
                    finalizeRecording()
                    isRecording = false
                    recordingTimer.stop()
                }
            }
        }

        Item {
            Layout.fillHeight: true
        }
    }

    function finalizeRecording() {
        if (navRoot && typeof navRoot.bumpSamples === "function" && gestureName.length > 0 && samplesRecorded > 0)
            navRoot.bumpSamples(gestureName, samplesRecorded)
    }

    Timer {
        id: recordingTimer
        interval: 800
        repeat: true
        onTriggered: {
            if (!isRecording)
                return
            if (samplesRecorded < targetSamples) {
                samplesRecorded++
            } else {
                stop()
                isRecording = false
                finalizeRecording()
            }
        }
    }
}
