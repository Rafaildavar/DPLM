import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 10. Настройки (камера, режим, алгоритм, порог, хранение).
 */
Item {
    id: root

    required property StackView gestStack

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

            Label {
                text: qsTr("Настройки системы (10)")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
                color: Material.foreground
            }
        }

        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: width
            contentHeight: settingsCol.height + 20
            clip: true

            ColumnLayout {
                id: settingsCol
                width: parent.width
                spacing: 16

                GroupBox {
                    title: qsTr("Камера")
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 8
                        width: parent.width

                        Label {
                            text: qsTr("Разрешение")
                        }
                        ComboBox {
                            Layout.fillWidth: true
                            model: ["640×480", "1280×720", "1920×1080"]
                        }

                        Label {
                            text: qsTr("Частота кадров, FPS")
                        }
                        SpinBox {
                            from: 15
                            to: 60
                            value: 30
                        }
                    }
                }

                GroupBox {
                    title: qsTr("Рабочий режим")
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 8
                        width: parent.width

                        RadioButton {
                            text: qsTr("Только распознавание")
                            checked: true
                        }
                        RadioButton {
                            text: qsTr("Обучение и распознавание")
                        }
                    }
                }

                GroupBox {
                    title: qsTr("Классификатор")
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 8
                        width: parent.width

                        Label {
                            text: qsTr("Алгоритм по умолчанию")
                        }
                        ComboBox {
                            id: clfAlgo
                            Layout.fillWidth: true
                            model: ["KNN", "SVM", "Авто"]
                        }

                        Label {
                            text: qsTr("Порог уверенности: ") + Math.round(thresholdSlider.value * 100) + "%"
                        }
                        Slider {
                            id: thresholdSlider
                            Layout.fillWidth: true
                            from: 0.5
                            to: 0.99
                            value: 0.75
                            stepSize: 0.01
                        }
                    }
                }

                GroupBox {
                    title: qsTr("Хранение данных")
                    Layout.fillWidth: true

                    ColumnLayout {
                        spacing: 8
                        width: parent.width

                        Label {
                            text: qsTr("Каталог датасета (заглушка)")
                            wrapMode: Text.WordWrap
                            font.pixelSize: 12
                            color: Material.color(Material.Grey, Material.Shade400)
                        }
                        TextField {
                            Layout.fillWidth: true
                            text: "~/.dplm/gestures"
                            readOnly: true
                        }

                        CheckBox {
                            text: qsTr("Автоочистка временных файлов")
                            checked: true
                        }
                    }
                }

                Label {
                    text: qsTr("Полная панель настроек также доступна по кнопке ⚙ в шапке главного окна.")
                    font.pixelSize: 11
                    color: Material.color(Material.Grey, Material.Shade500)
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }
    }
}
