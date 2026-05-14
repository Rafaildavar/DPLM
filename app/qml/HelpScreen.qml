import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 11. Справка — структурированные блоки, сценарии и советы по CV.
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    readonly property color cardColor: "#232336"
    readonly property color cardBorder: "#3d3d55"

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        RowLayout {
            Layout.fillWidth: true
            Layout.margins: 4
            spacing: 8

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2

                Label {
                    text: qsTr("Справка")
                    font.pixelSize: 20
                    font.bold: true
                    color: Material.foreground
                }

                Label {
                    text: qsTr("DPLM — жесты, камера и сценарии дипломного питча")
                    font.pixelSize: 12
                    color: Material.color(Material.Grey, Material.Shade400)
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                }
            }
        }

        ScrollView {
            id: helpScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.topMargin: 4
            clip: true

            ColumnLayout {
                id: helpCol
                width: helpScroll.availableWidth
                spacing: 14

                // Карточка: о проекте
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardColor
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: aboutCol.implicitHeight + 28

                    RowLayout {
                        id: aboutCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.margins: 14
                        spacing: 12

                        Rectangle {
                            Layout.preferredWidth: 44
                            Layout.preferredHeight: 44
                            radius: 10
                            color: "#2a3d52"

                            Label {
                                anchors.centerIn: parent
                                text: "📘"
                                font.pixelSize: 22
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: qsTr("Что это за приложение")
                                font.bold: true
                                font.pixelSize: 15
                                color: Material.accent
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                font.pixelSize: 13
                                color: Material.foreground
                                text: qsTr("DPLM объединяет интерфейс на Qt/QML, захват камеры (OpenCV) и конвейер жестов: MediaPipe для ключевых точек кисти, обучаемый классификатор (KNN в models/) и привязку жестов к системным командам.")
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardColor
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: flowCol.implicitHeight + 28

                    ColumnLayout {
                        id: flowCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10

                        Label {
                            text: qsTr("Типовой поток работы")
                            font.bold: true
                            font.pixelSize: 15
                            color: Material.accent
                        }

                        Repeater {
                            model: [
                                qsTr("Включите камеру на главном экране модуля «Жесты», если нужен живой кадр."),
                                qsTr("Создайте класс жеста и запишите ≥ 20 примеров с разным наклоном кисти и расстоянием до камеры."),
                                qsTr("Привяжите жест к команде из реестра, при необходимости обучите или сравните модели."),
                                qsTr("Запустите распознавание: встроенное на экране «Распознавание» или фоновый режим снизу главного окна (cv/realtime_infer.py).")
                            ]

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.minimumHeight: 32
                                spacing: 10

                                Rectangle {
                                    Layout.preferredWidth: 26
                                    Layout.preferredHeight: 26
                                    radius: 13
                                    color: Material.accent

                                    Label {
                                        anchors.centerIn: parent
                                        text: (index + 1).toString()
                                        font.bold: true
                                        font.pixelSize: 12
                                        color: "#1a1a2e"
                                    }
                                }

                                Label {
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                    font.pixelSize: 13
                                    color: Material.foreground
                                    text: modelData
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardColor
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: recCol.implicitHeight + 28

                    ColumnLayout {
                        id: recCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 8

                        Label {
                            text: qsTr("Запись примеров")
                            font.bold: true
                            font.pixelSize: 15
                            color: Material.accent
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            font.pixelSize: 13
                            color: Material.foreground
                            text: qsTr("• Делайте паузу между образцами; слегка меняйте угол ладони и положение в кадре.\n"
                                       + "• Не уводите руку резко, пока идёт запись очередного образца.\n"
                                       + "• Равномерный свет без контровой тени на кисти; избегайте яркого окна за спиной.\n"
                                       + "• Спокойный фон без движущихся объектов позади руки улучшает стабильность точек.")
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardColor
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: modeCol.implicitHeight + 28

                    ColumnLayout {
                        id: modeCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 8

                        Label {
                            text: qsTr("Режимы и вкладки")
                            font.bold: true
                            font.pixelSize: 15
                            color: Material.accent
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            font.pixelSize: 13
                            color: Material.foreground
                            text: qsTr("«Жесты» — основной стек из 12 экранов.\n"
                                       + "«Питч» — тот же набор экранов в виде списка слева для демонстрации диплома.\n"
                                       + "«Помощник» — голосовые сценарии (требуются зависимости вроде PyAudio/Vosk).\n"
                                       + "«Команды» — просмотр и управление привязками команд.")
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: Qt.rgba(0.2, 0.55, 0.45, 0.15)
                    border.width: 1
                    border.color: Qt.rgba(0.3, 0.85, 0.65, 0.45)
                    implicitHeight: tipCol.implicitHeight + 28

                    ColumnLayout {
                        id: tipCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 6

                        Label {
                            text: qsTr("Если что-то не работает")
                            font.bold: true
                            font.pixelSize: 15
                            color: "#7FFFD4"
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            font.pixelSize: 13
                            color: Material.foreground
                            text: qsTr("• Нет превью — проверьте доступ к камере в macOS «Конфиденциальность».\n"
                                       + "• MediaPipe: при новых версиях пакета используется Tasks API; при первом запуске может скачаться hand_landmarker.task.\n"
                                       + "• Нет knn.pkl — обучите модель и положите файл в models/ (см. скрипты в cv/).")
                        }
                    }
                }

                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 20
                }
            }
        }
    }
}
