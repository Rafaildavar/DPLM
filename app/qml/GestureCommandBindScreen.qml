import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 6. Привязка жестов к командам.
 */
Item {
    id: root

    required property StackView gestStack
    property var gestureCatalog
    property var pitchHost

    ListModel {
        id: commandsModel
        Component.onCompleted: {
            var cmds = appController.getAvailableCommands()
            for (var i = 0; i < cmds.length; i++) {
                var c = cmds[i]
                commandsModel.append({
                                        "name": c.name || c.title || ("cmd_" + i)
                                    })
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            Label {
                text: qsTr("Привязка жестов к командам (6)")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
                color: Material.foreground
            }
        }

        Label {
            text: qsTr("Жест из словаря")
            font.bold: true
            color: Material.foreground
        }

        ComboBox {
            id: gesturePick
            Layout.fillWidth: true
            model: gestureCatalog
            textRole: "name"
            enabled: gestureCatalog && gestureCatalog.count > 0
        }

        Label {
            text: qsTr("Команда")
            font.bold: true
            color: Material.foreground
        }

        ComboBox {
            id: commandPick
            Layout.fillWidth: true
            model: commandsModel
            textRole: "name"
        }

        Button {
            Layout.fillWidth: true
            text: qsTr("Сохранить привязку")
            highlighted: true
            onClicked: {
                if (gesturePick.currentIndex < 0 || commandPick.currentIndex < 0)
                    return
                bindStatus.text = qsTr("Сохранено: «") + gesturePick.currentText + qsTr("» → ") + commandPick.currentText
            }
        }

        Label {
            id: bindStatus
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            font.pixelSize: 13
            color: Material.accent
            text: qsTr("Измените выбор и нажмите «Сохранить привязку».")
        }

        Item {
            Layout.fillHeight: true
        }
    }
}
