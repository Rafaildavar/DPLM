import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 12. Тестирование жеста и команды.
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    property string lastGesture: "—"
    property string mappedCommand: "—"

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
                text: qsTr("Тестирование (12)")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
                color: Material.foreground
            }
        }

        Label {
            text: qsTr("Пробный прогон: имитация распознавания и привязки к команде (демо).")
            font.pixelSize: 12
            color: Material.color(Material.Grey, Material.Shade500)
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }

        CameraPreview {
            Layout.fillWidth: true
            Layout.preferredHeight: 160
            previewHeight: 160
            cornerRadius: 14
        }

        Button {
            Layout.fillWidth: true
            text: qsTr("Выполнить пробную проверку")
            highlighted: true
            onClicked: {
                var gestures = [qsTr("Свайп вправо"), qsTr("Большой палец"), qsTr("Ладонь")]
                var cmds = ["Open Browser", "Volume Up", "Close Window"]
                var i = Math.floor(Math.random() * 3)
                lastGesture = gestures[i]
                mappedCommand = cmds[i]
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 100
            radius: 10
            color: "#2a2a3e"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 6

                Label {
                    text: qsTr("Распознанный жест (тест): ") + lastGesture
                    color: Material.foreground
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
                Label {
                    text: qsTr("Будет выполнена команда: ") + mappedCommand
                    color: Material.accent
                    font.bold: true
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }

        Item {
            Layout.fillHeight: true
        }
    }
}
