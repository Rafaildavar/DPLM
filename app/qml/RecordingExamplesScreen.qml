import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 4. Запись примеров: камера, осмысленное добавление образцов по кнопке,
 * прогресс, подсказки (без фиктивного таймера +1).
 */
Item {
    id: root

    required property StackView gestStack
    property string gestureName: ""
    property var navRoot
    property var pitchHost
    property var gestureCatalog

    property int samplesRecorded: 0
    readonly property int targetSamples: 20
    /** Камера была запущена именно с этого экрана — выключим при уходе */
    property bool cameraStartedHere: false
    property int lastCaptureMs: 0
    readonly property int cooldownMs: 500
    readonly property bool hasClassName: gestureName.trim().length > 0
    readonly property bool quotaDone: samplesRecorded >= targetSamples

    StackView.onStatusChanged: {
        if (StackView.status === StackView.Active) {
            if ((!gestureName || gestureName.trim().length === 0) && gestureCatalog && gestureCatalog.count > 0) {
                var first = gestureCatalog.get(0)
                if (first && first.name)
                    gestureName = String(first.name).trim()
            }
            if (pitchHost && pitchHost.setPitchScreenIndex)
                pitchHost.setPitchScreenIndex(3)
            if (!appController.isCameraActive) {
                appController.startCamera()
                cameraStartedHere = true
            }
        } else if (StackView.status === StackView.Inactive || StackView.status === StackView.Deactivating) {
            if (cameraStartedHere) {
                appController.stopCamera()
                cameraStartedHere = false
            }
        }
    }

    function tryCaptureSample() {
        if (!hasClassName || quotaDone)
            return
        var now = Date.now()
        if (now - lastCaptureMs < cooldownMs)
            return
        lastCaptureMs = now
        if (samplesRecorded === 0)
            appController.startGestureTraining(gestureName)
        samplesRecorded += 1
        appController.logGestureSample(gestureName, samplesRecorded)
        if (samplesRecorded >= targetSamples)
            finalizeRecording()
    }

    function finalizeRecording() {
        if (navRoot && typeof navRoot.bumpSamples === "function" && gestureName.length > 0 && samplesRecorded > 0)
            navRoot.bumpSamples(gestureName, samplesRecorded)
    }

    ScrollView {
        id: scr
        anchors.fill: parent
        anchors.margins: 8
        clip: true

        ColumnLayout {
            width: scr.availableWidth
            spacing: 14

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                ToolButton {
                    text: "←"
                    font.pixelSize: 18
                    onClicked: {
                        finalizeRecording()
                        if (pitchHost && pitchHost.handleBack)
                            pitchHost.handleBack(gestStack)
                        else
                            gestStack.pop()
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    Label {
                        text: qsTr("Запись примеров")
                        font.pixelSize: 20
                        font.bold: true
                        color: Material.foreground
                    }

                    Label {
                        text: hasClassName ? (qsTr("Класс: ") + gestureName) : qsTr("Сначала задайте имя класса на экране «Новый жест»")
                        font.pixelSize: 13
                        font.bold: hasClassName
                        color: hasClassName ? Material.accent : Material.color(Material.Orange, Material.Shade200)
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
            }

        Rectangle {
            Layout.fillWidth: true
            radius: 10
            color: "#232336"
            border.width: 1
            border.color: hasClassName ? "#3d3d55" : Material.color(Material.Orange, Material.Shade300)
            implicitHeight: classRow.implicitHeight + 16

            RowLayout {
                id: classRow
                anchors.fill: parent
                anchors.margins: 8
                spacing: 10

                Label {
                    text: qsTr("Класс")
                    font.pixelSize: 12
                    color: Material.color(Material.Grey, Material.Shade400)
                }

                TextField {
                    id: classField
                    Layout.fillWidth: true
                    placeholderText: qsTr("Например: wave_hello")
                    text: root.gestureName
                    onTextChanged: root.gestureName = text.trim()
                }
            }
        }

            // Статус камеры
            Rectangle {
                Layout.fillWidth: true
                radius: 10
                color: appController.isCameraActive ? Qt.rgba(0.1, 0.45, 0.35, 0.35) : Qt.rgba(0.45, 0.25, 0.1, 0.35)
                border.width: 1
                border.color: appController.isCameraActive ? "#2E7D6A" : "#8D6E63"
                implicitHeight: camRow.implicitHeight + 16

                RowLayout {
                    id: camRow
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 10

                    Label {
                        text: appController.isCameraActive ? "●" : "○"
                        font.pixelSize: 14
                        color: appController.isCameraActive ? "#69F0AE" : "#FFCC80"
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        font.pixelSize: 12
                        color: Material.foreground
                        text: appController.isCameraActive
                              ? qsTr("Камера активна — покажите жест и нажмите «Зафиксировать образец» после каждого положения руки.")
                              : qsTr("Камера не запущена. Проверьте разрешения macOS → Camera или кнопку «Запустить камеру» на главном экране «Жесты».")
                    }
                }
            }

            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: videoBox.height + 8

                Item {
                    id: videoBox
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: Math.min(scr.availableWidth - 16, 560)
                    height: Math.round(width * 9 / 16)
                    clip: true

                    CameraPreview {
                        anchors.fill: parent
                        previewHeight: height
                        cornerRadius: 16
                    }

                    Rectangle {
                        anchors.fill: parent
                        radius: 16
                        color: "transparent"
                        border.width: quotaDone ? 3 : 0
                        border.color: "#69F0AE"
                    }
                }
            }

            // Счётчик и прогресс
            Rectangle {
                Layout.fillWidth: true
                Layout.maximumWidth: 640
                Layout.alignment: Qt.AlignHCenter
                radius: 14
                color: "#232336"
                border.width: 1
                border.color: "#3d3d55"
                implicitHeight: progCol.implicitHeight + 20

                ColumnLayout {
                    id: progCol
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true

                        Label {
                            text: qsTr("Образцов")
                            font.pixelSize: 12
                            color: Material.color(Material.Grey, Material.Shade400)
                        }

                        Item {
                            Layout.fillWidth: true
                        }

                        Label {
                            text: samplesRecorded + " / " + targetSamples
                            font.bold: true
                            font.pixelSize: 22
                            color: quotaDone ? "#69F0AE" : Material.accent
                        }
                    }

                    ProgressBar {
                        Layout.fillWidth: true
                        from: 0
                        to: targetSamples
                        value: samplesRecorded
                    }

                    Label {
                        visible: quotaDone
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        font.pixelSize: 12
                        color: "#69F0AE"
                        text: qsTr("Цель достигнута. Можно вернуться назад — счётчик в питче обновится.")
                    }
                }
            }

            Button {
                Layout.fillWidth: true
                Layout.maximumWidth: 640
                Layout.alignment: Qt.AlignHCenter
                visible: !appController.isCameraActive
                text: qsTr("Включить камеру")
                onClicked: {
                    appController.startCamera()
                    cameraStartedHere = true
                }
            }

            // Главное действие — одно нажатие = один осмысленный образец
            Button {
                Layout.fillWidth: true
                Layout.maximumWidth: 640
                Layout.alignment: Qt.AlignHCenter
                Layout.preferredHeight: 48
                highlighted: true
                enabled: hasClassName && !quotaDone && appController.isCameraActive
                text: quotaDone ? qsTr("Лимит образцов достигнут") : qsTr("Зафиксировать образец (+1)")
                onClicked: tryCaptureSample()
            }

            Label {
                Layout.fillWidth: true
                Layout.maximumWidth: 640
                Layout.alignment: Qt.AlignHCenter
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                font.pixelSize: 11
                color: Material.color(Material.Grey, Material.Shade500)
                text: qsTr("Каждое нажатие — отдельный образец (не автоматическая «стрельба» по таймеру). Меняйте положение руки между нажатиями.")
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.maximumWidth: 640
                Layout.alignment: Qt.AlignHCenter
                spacing: 10

                Button {
                    Layout.fillWidth: true
                    text: qsTr("Сбросить счётчик")
                    flat: true
                    enabled: samplesRecorded > 0 && !quotaDone
                    onClicked: samplesRecorded = 0
                }

                Button {
                    Layout.fillWidth: true
                    text: qsTr("Завершить и сохранить в питче")
                    enabled: samplesRecorded > 0
                    onClicked: {
                        finalizeRecording()
                        if (pitchHost && pitchHost.handleBack)
                            pitchHost.handleBack(gestStack)
                        else
                            gestStack.pop()
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.maximumWidth: 640
                Layout.alignment: Qt.AlignHCenter
                radius: 12
                color: "#243040"
                implicitHeight: hintCol.implicitHeight + 24

                Column {
                    id: hintCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.margins: 12
                    spacing: 8

                    Label {
                        width: parent.width
                        text: qsTr("Подсказка")
                        font.bold: true
                        font.pixelSize: 12
                        color: Material.accent
                    }

                    Label {
                        width: parent.width
                        wrapMode: Text.WordWrap
                        font.pixelSize: 12
                        color: Material.foreground
                        text: qsTr("Полный пайплайн записи кадров в .npy — скрипт cv/record_gestures.py. Здесь фиксируются осмысленные нажатия и журнал samples_log.jsonl в data/gestures/<класс>/ для демонстрации потока.")
                    }
                }
            }

            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: 24
            }
        }
    }
}
