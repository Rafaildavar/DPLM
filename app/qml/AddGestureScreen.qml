import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 3. Добавление нового жеста: имя, создание класса, переход к записи примеров.
 */
Item {
    id: root

    StackView.onStatusChanged: {
        if (StackView.status === StackView.Active && pitchHost && pitchHost.setPitchScreenIndex)
            pitchHost.setPitchScreenIndex(2)
    }

    required property StackView gestStack
    property var gestureCatalog
    property var navRoot
    property var pitchHost

    ColumnLayout {
        anchors.fill: parent
        spacing: 14

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            Text {
                text: qsTr("Новый жест (3)")
                font.pixelSize: 18
                font.bold: true
                color: Material.foreground
                Layout.fillWidth: true
            }
        }

        Label {
            text: qsTr("Название жеста")
            font.bold: true
            color: Material.foreground
        }

        TextField {
            id: gestureNameField
            Layout.fillWidth: true
            placeholderText: qsTr("Например: wave_hello")
        }

        Button {
            Layout.fillWidth: true
            text: qsTr("Создать новый класс")
            highlighted: true
            onClicked: {
                if (gestureNameField.text.trim().length === 0)
                    return
                if (gestureCatalog) {
                    gestureCatalog.append({
                                            "name": gestureNameField.text.trim(),
                                            "samples": 0,
                                            "readyStr": qsTr("Нет примеров")
                                        })
                }
            }
        }

        Button {
            Layout.fillWidth: true
            text: qsTr("Записать обучающие примеры")
            onClicked: {
                if (gestureNameField.text.trim().length === 0)
                    return
                gestStack.push(Qt.resolvedUrl("RecordingExamplesScreen.qml"), {
                                   "gestStack": gestStack,
                                   "gestureCatalog": gestureCatalog,
                                   "gestureName": gestureNameField.text.trim(),
                                   "navRoot": navRoot,
                                   "pitchHost": pitchHost
                               })
            }
        }

        Item {
            Layout.fillHeight: true
        }
    }
}
