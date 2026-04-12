import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 1. Главный: видеопоток, камера, обучение, распознавание, статус;
 * меню переходов на экраны 2–12.
 */
Item {
    id: root

    required property StackView gestStack
    property var gestureCatalog
    property var navRoot
    property var pitchHost

    function navArgs(extra) {
        return Object.assign({
                                 gestStack: gestStack,
                                 gestureCatalog: gestureCatalog,
                                 navRoot: navRoot,
                                 pitchHost: pitchHost
                             }, extra || {})
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        Text {
            text: qsTr("Главный экран (1)")
            font.pixelSize: 18
            font.bold: true
            color: Material.foreground
        }

        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: width
            contentHeight: homeCol.height + 20
            clip: true

            Column {
                id: homeCol
                width: parent.width
                spacing: 12

                CameraPreview {
                    width: parent.width
                    height: 200
                    previewHeight: 200
                }

                Button {
                    width: parent.width
                    text: appController.isCameraActive ? qsTr("Остановить камеру") : qsTr("Запустить камеру")
                    highlighted: true
                    onClicked: {
                        if (appController.isCameraActive)
                            appController.stopCamera()
                        else
                            appController.startCamera()
                    }
                }

                RowLayout {
                    width: parent.width
                    spacing: 8

                    Button {
                        Layout.fillWidth: true
                        text: qsTr("Режим обучения")
                        onClicked: gestStack.push(Qt.resolvedUrl("AddGestureScreen.qml"), root.navArgs({}))
                    }

                    Button {
                        Layout.fillWidth: true
                        text: qsTr("Режим распознавания")
                        onClicked: gestStack.push(Qt.resolvedUrl("GestureRecognitionScreen.qml"), root.navArgs({}))
                    }
                }

                Label {
                    width: parent.width
                    text: qsTr("Все экраны системы (по номерам)")
                    font.bold: true
                    color: Material.color(Material.Grey, Material.Shade400)
                }

                Button {
                    width: parent.width
                    text: qsTr("2. Распознавание жестов")
                    onClicked: gestStack.push(Qt.resolvedUrl("GestureRecognitionScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("3. Добавление нового жеста")
                    onClicked: gestStack.push(Qt.resolvedUrl("AddGestureScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("4. Запись обучающих примеров")
                    onClicked: gestStack.push(Qt.resolvedUrl("RecordingExamplesScreen.qml"), root.navArgs({ "gestureName": "" }))
                }

                Button {
                    width: parent.width
                    text: qsTr("5. Список жестов")
                    onClicked: gestStack.push(Qt.resolvedUrl("GestureListScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("6. Привязка жестов к командам")
                    flat: true
                    onClicked: gestStack.push(Qt.resolvedUrl("GestureCommandBindScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("7. Обучение модели")
                    flat: true
                    onClicked: gestStack.push(Qt.resolvedUrl("ModelTrainingScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("8. Сравнение моделей")
                    flat: true
                    onClicked: gestStack.push(Qt.resolvedUrl("ModelCompareScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("9. Сообщения и ошибки")
                    flat: true
                    onClicked: gestStack.push(Qt.resolvedUrl("MessagesLogScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("10. Настройки системы")
                    flat: true
                    onClicked: gestStack.push(Qt.resolvedUrl("SystemSettingsScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("11. Справка")
                    flat: true
                    onClicked: gestStack.push(Qt.resolvedUrl("HelpScreen.qml"), root.navArgs({}))
                }

                Button {
                    width: parent.width
                    text: qsTr("12. Тестирование")
                    flat: true
                    onClicked: gestStack.push(Qt.resolvedUrl("GestureTestScreen.qml"), root.navArgs({}))
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 56
            radius: 12
            color: "#2a2a3e"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 4

                Text {
                    text: qsTr("Текущий статус")
                    font.pixelSize: 11
                    color: Material.color(Material.Grey, Material.Shade500)
                }

                Text {
                    text: appController.isCameraActive
                          ? (appController.isRecognizing ? qsTr("Камера: вкл. · Распознавание активно") : qsTr("Камера: вкл. · превью OpenCV"))
                          : qsTr("Камера: выкл. · ") + appController.status
                    font.pixelSize: 13
                    color: Material.foreground
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }
    }
}
