import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 5. Список жестов: карточки, прогресс набора примеров, действия.
 */
Item {
    id: root

    required property StackView gestStack
    property var gestureCatalog
    property var navRoot
    property var pitchHost

    readonly property int samplesTarget: 20

    function gestureInitial(name) {
        var s = (name || "").trim()
        if (!s.length)
            return "?"
        return s.charAt(0).toUpperCase()
    }

    function progress01(samples) {
        return Math.min(1, Math.max(0, Number(samples) / samplesTarget))
    }

    function isReady(samples) {
        return Number(samples) >= samplesTarget
    }

    Dialog {
        id: deleteConfirmDialog
        property int rowIndex: -1
        property string gestureName: ""

        title: qsTr("Удалить класс жеста?")
        modal: true
        standardButtons: Dialog.No | Dialog.Yes
        anchors.centerIn: Overlay.overlay
        width: Math.min(400, Overlay.overlay ? Overlay.overlay.width - 48 : 400)

        onAccepted: {
            if (gestureCatalog && rowIndex >= 0 && rowIndex < gestureCatalog.count)
                gestureCatalog.remove(rowIndex)
            rowIndex = -1
            gestureName = ""
        }
        onRejected: {
            rowIndex = -1
            gestureName = ""
        }

        Label {
            text: deleteConfirmDialog.gestureName.length
                  ? qsTr("Будет удалена запись «%1». Это действие нельзя отменить.").arg(deleteConfirmDialog.gestureName)
                  : qsTr("Удалить выбранную запись?")
            wrapMode: Text.WordWrap
            width: parent.width
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2

                Label {
                    text: qsTr("Список жестов")
                    font.pixelSize: 20
                    font.bold: true
                    color: Material.foreground
                }
                Label {
                    text: qsTr("Классы для обучения и распознавания")
                    font.pixelSize: 12
                    color: Material.color(Material.Grey, Material.Shade500)
                }
            }

            Rectangle {
                visible: gestureCatalog && gestureCatalog.count > 0
                radius: 12
                color: Material.color(Material.Cyan, Material.Shade900)
                border.width: 1
                border.color: Qt.rgba(Material.accent.r, Material.accent.g, Material.accent.b, 0.35)
                implicitWidth: countBadge.implicitWidth + 20
                implicitHeight: 32

                Label {
                    id: countBadge
                    anchors.centerIn: parent
                    text: gestureCatalog ? gestureCatalog.count : 0
                    font.pixelSize: 14
                    font.bold: true
                    color: Material.accent
                }
            }
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ScrollView {
                id: scroll
                anchors.fill: parent
                visible: gestureCatalog && gestureCatalog.count > 0
                clip: true
                ScrollBar.vertical.policy: ScrollBar.AsNeeded

                ListView {
                    id: listView
                    width: scroll.availableWidth
                    implicitHeight: contentHeight
                    spacing: 10
                    clip: true
                    model: gestureCatalog
                    boundsBehavior: Flickable.StopAtBounds

                    delegate: Rectangle {
                    id: card
                    width: listView.width
                    implicitHeight: cardLayout.implicitHeight + 24
                    radius: 14
                    color: cardMa.containsMouse ? "#32324a" : "#262636"
                    border.width: 1
                    border.color: cardMa.containsMouse
                                  ? Qt.rgba(Material.accent.r, Material.accent.g, Material.accent.b, 0.45)
                                  : Material.color(Material.Grey, Material.Shade800)

                    Behavior on color {
                        ColorAnimation {
                            duration: 120
                        }
                    }

                    Behavior on border.color {
                        ColorAnimation {
                            duration: 120
                        }
                    }

                    MouseArea {
                        id: cardMa
                        anchors.fill: parent
                        hoverEnabled: true
                        acceptedButtons: Qt.NoButton
                    }

                    ColumnLayout {
                        id: cardLayout
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.leftMargin: 14
                        anchors.rightMargin: 12
                        spacing: 10

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 14

                            Rectangle {
                                Layout.preferredWidth: 48
                                Layout.preferredHeight: 48
                                radius: 24
                                gradient: Gradient {
                                    GradientStop {
                                        position: 0
                                        color: Qt.lighter(Material.accent, 1.15)
                                    }
                                    GradientStop {
                                        position: 1
                                        color: Qt.darker(Material.accent, 1.35)
                                    }
                                }

                                Label {
                                    anchors.centerIn: parent
                                    text: gestureInitial(model.name)
                                    font.pixelSize: 20
                                    font.bold: true
                                    color: "#1a1a2e"
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 6

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Label {
                                        text: model.name
                                        font.pixelSize: 16
                                        font.bold: true
                                        color: Material.foreground
                                        wrapMode: Text.WordWrap
                                        Layout.fillWidth: true
                                    }

                                    Rectangle {
                                        radius: 10
                                        color: isReady(model.samples)
                                               ? Qt.rgba(0.3, 0.75, 0.45, 0.22)
                                               : Material.color(Material.Grey, Material.Shade800)
                                        implicitHeight: statusPill.implicitHeight + 8
                                        implicitWidth: statusPill.implicitWidth + 16

                                        Label {
                                            id: statusPill
                                            anchors.centerIn: parent
                                            text: isReady(model.samples) ? qsTr("Готово") : qsTr("Набор")
                                            font.pixelSize: 11
                                            font.bold: true
                                            color: isReady(model.samples)
                                                   ? "#A5D6A7"
                                                   : Material.color(Material.Grey, Material.Shade400)
                                        }
                                    }
                                }

                                Label {
                                    text: model.readyStr
                                    font.pixelSize: 12
                                    color: Material.color(Material.Grey, Material.Shade400)
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 4

                                    RowLayout {
                                        Layout.fillWidth: true

                                        Label {
                                            text: qsTr("Примеры")
                                            font.pixelSize: 10
                                            color: Material.color(Material.Grey, Material.Shade500)
                                        }
                                        Item {
                                            Layout.fillWidth: true
                                        }
                                        Label {
                                            text: model.samples + " / " + samplesTarget
                                            font.pixelSize: 10
                                            font.bold: true
                                            color: Material.color(Material.Grey, Material.Shade400)
                                        }
                                    }

                                    ProgressBar {
                                        Layout.fillWidth: true
                                        from: 0
                                        to: 1
                                        value: progress01(model.samples)
                                        palette.accent: isReady(model.samples)
                                                        ? "#66BB6A"
                                                        : Material.accent
                                    }
                                }
                            }

                            ColumnLayout {
                                spacing: 6

                                RoundButton {
                                    text: "✎"
                                    font.pixelSize: 15
                                    Material.background: Material.color(Material.Grey, Material.Shade700)
                                    implicitWidth: 40
                                    implicitHeight: 40

                                    ToolTip.visible: hovered
                                    ToolTip.text: qsTr("Демо: пометить имя (для питча)")
                                    ToolTip.delay: 500

                                    onClicked: gestureCatalog.setProperty(index, "name", model.name + qsTr(" ·"))
                                }

                                RoundButton {
                                    text: "🗑"
                                    font.pixelSize: 14
                                    Material.background: Material.color(Material.Red, Material.Shade900)
                                    implicitWidth: 40
                                    implicitHeight: 40

                                    ToolTip.visible: hovered
                                    ToolTip.text: qsTr("Удалить класс")
                                    ToolTip.delay: 500

                                    onClicked: {
                                        deleteConfirmDialog.rowIndex = index
                                        deleteConfirmDialog.gestureName = model.name
                                        deleteConfirmDialog.open()
                                    }
                                }
                            }
                        }
                    }
                }
            }
            }

            Rectangle {
            anchors.fill: parent
            visible: !gestureCatalog || gestureCatalog.count === 0
            radius: 14
            color: "#222230"
            border.width: 1
            border.color: Material.color(Material.Grey, Material.Shade800)

            ColumnLayout {
                id: emptyStateLayout
                anchors.centerIn: parent
                width: parent.width - 40
                spacing: 10

                Label {
                    text: "◇"
                    font.pixelSize: 28
                    color: Material.color(Material.Grey, Material.Shade600)
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: qsTr("Классов жестов пока нет")
                    font.pixelSize: 16
                    font.bold: true
                    color: Material.foreground
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: qsTr("Создайте первый жест на экране «Добавление жеста» (3), затем вернитесь сюда.")
                    font.pixelSize: 12
                    color: Material.color(Material.Grey, Material.Shade500)
                    wrapMode: Text.WordWrap
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
            }
            }
        }
    }
}
