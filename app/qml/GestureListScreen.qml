import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 5. Список жестов: примеры, готовность, редактирование, удаление.
 */
Item {
    id: root

    required property StackView gestStack
    property var gestureCatalog

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: gestStack.pop()
            }

            Text {
                text: qsTr("Список жестов (5)")
                font.pixelSize: 18
                font.bold: true
                color: Material.foreground
                Layout.fillWidth: true
            }
        }

        ListView {
            id: listView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 8
            model: gestureCatalog

            delegate: Rectangle {
                width: listView.width
                height: rowInner.height + 20
                radius: 10
                color: "#2a2a3e"

                ColumnLayout {
                    id: rowInner
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 10
                    spacing: 6

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Text {
                            text: "✋"
                            font.pixelSize: 20
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2

                            Label {
                                text: model.name
                                font.pixelSize: 15
                                font.bold: true
                                color: Material.foreground
                            }

                            Label {
                                text: qsTr("Примеров: ") + model.samples + qsTr(" · ") + model.readyStr
                                font.pixelSize: 12
                                color: Material.color(Material.Grey, Material.Shade400)
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }

                        RowLayout {
                            spacing: 6

                            ToolButton {
                                text: "✎"
                                font.pixelSize: 14
                                hoverEnabled: true
                                onClicked: {
                                    gestureCatalog.setProperty(index, "name", model.name + qsTr(" ·"))
                                }
                            }

                            ToolButton {
                                text: "🗑"
                                font.pixelSize: 14
                                hoverEnabled: true
                                onClicked: gestureCatalog.remove(index)
                            }
                        }
                    }
                }
            }
        }

        Label {
            visible: !gestureCatalog || gestureCatalog.count === 0
            text: qsTr("Классов нет. Создайте на экране 3.")
            wrapMode: Text.WordWrap
            font.pixelSize: 12
            color: Material.color(Material.Grey, Material.Shade500)
            Layout.fillWidth: true
        }
    }
}
