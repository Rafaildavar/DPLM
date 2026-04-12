import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 9. Сообщения и ошибки.
 */
Item {
    id: root

    required property StackView gestStack

    ListModel {
        id: logModel
        ListElement {
            time: "12:04"
            level: "INFO"
            text: "Модель загружена: gesture_clf.pkl"
        }
        ListElement {
            time: "12:05"
            level: "WARN"
            text: "Рука не обнаружена в кадре (таймаут 2 с)"
        }
        ListElement {
            time: "12:06"
            level: "ERR"
            text: "Камера: доступ запрещён (проверьте разрешения ОС)"
        }
        ListElement {
            time: "12:07"
            level: "INFO"
            text: "Недостаточно примеров для класса «wave»: 8/20"
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: gestStack.pop()
            }

            Label {
                text: qsTr("Сообщения и ошибки (9)")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
                color: Material.foreground
            }

            ToolButton {
                text: "↻"
                font.pixelSize: 16
                onClicked: logModel.append({
                                                "time": Qt.formatTime(new Date(), "hh:mm"),
                                                "level": "INFO",
                                                "text": qsTr("Системное уведомление: проверка связи")
                                            })
            }
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 6
            model: logModel

            delegate: Rectangle {
                width: ListView.view.width
                height: msgCol.height + 16
                radius: 8
                color: "#2a2a3e"

                ColumnLayout {
                    id: msgCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 8
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label {
                            text: model.time
                            font.pixelSize: 11
                            color: Material.color(Material.Grey, Material.Shade500)
                        }

                        Label {
                            text: model.level
                            font.bold: true
                            font.pixelSize: 11
                            color: model.level === "ERR" ? "#f44336" : (model.level === "WARN" ? "#FF9800" : Material.accent)
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        text: model.text
                        wrapMode: Text.WordWrap
                        font.pixelSize: 13
                        color: Material.foreground
                    }
                }
            }
        }
    }
}
