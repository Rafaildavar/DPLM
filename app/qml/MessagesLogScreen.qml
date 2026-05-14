import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 9. Сообщения и ошибки — фильтр, поиск, карточки, статистика (демо-модель).
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    readonly property color cardBg: "#232336"
    readonly property color cardBorder: "#3d3d55"

    property string filterTag: "ALL"
    /** Инкремент при изменении fullLogModel — чтобы пересчитать сводку ERR/WARN/INFO */
    property int statNonce: 0

    ListModel {
        id: fullLogModel
    }

    ListModel {
        id: displayModel
    }

    function seedDemo() {
        fullLogModel.clear()
        var rows = [
            {
                time: "09:12",
                level: "INFO",
                text: qsTr("Приложение запущено, Qt Material theme активен.")
            },
            {
                time: "09:14",
                level: "INFO",
                text: qsTr("Камера: поток превью OpenCV (1280×720, AVFoundation).")
            },
            {
                time: "09:18",
                level: "WARN",
                text: qsTr("PyAudio не найден — голосовой ввод недоступен до установки пакета.")
            },
            {
                time: "09:21",
                level: "INFO",
                text: qsTr("Встроенное CV: Hand Landmarker готов, модель KNN при наличии models/knn.pkl.")
            },
            {
                time: "09:24",
                level: "WARN",
                text: qsTr("Рука не в кадре > 2 с — пропуск кадра классификации.")
            },
            {
                time: "09:27",
                level: "ERR",
                text: qsTr("Камера: доступ запрещён в настройках конфиденциальности macOS.")
            },
            {
                time: "09:31",
                level: "INFO",
                text: qsTr("Запись примеров: класс «wave», каталог data/gestures/wave/ создан.")
            },
            {
                time: "09:35",
                level: "WARN",
                text: qsTr("sklearn: предупреждение о версии при загрузке knn.pkl — при необходимости переобучите модель.")
            },
            {
                time: "09:40",
                level: "INFO",
                text: qsTr("Фоновое распознавание остановлено (SIGTERM процессу realtime_infer).")
            }
        ]
        for (var i = 0; i < rows.length; i++)
            fullLogModel.append(rows[i])
        statNonce++
        applyFilter()
    }

    function applyFilter() {
        displayModel.clear()
        var q = searchField.text.toLowerCase().trim()
        for (var i = 0; i < fullLogModel.count; i++) {
            var row = fullLogModel.get(i)
            if (filterTag !== "ALL" && row.level !== filterTag)
                continue
            if (q.length > 0 && String(row.text).toLowerCase().indexOf(q) < 0)
                continue
            displayModel.append({
                                    "time": row.time,
                                    "level": row.level,
                                    "text": row.text
                                })
        }
    }

    function countLevel(lvl) {
        var n = 0
        for (var i = 0; i < fullLogModel.count; i++) {
            if (fullLogModel.get(i).level === lvl)
                n++
        }
        return n
    }

    function appendLiveLine(level, text) {
        var t = Qt.formatTime(new Date(), "hh:mm:ss")
        fullLogModel.append({
                                "time": t,
                                "level": level,
                                "text": text
                            })
        statNonce++
        applyFilter()
    }

    Component.onCompleted: seedDemo()

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
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
                    text: qsTr("Сообщения и ошибки")
                    font.pixelSize: 20
                    font.bold: true
                    color: Material.foreground
                }

                Label {
                    text: qsTr("Журнал событий UI и подсистем (демо). В продукте — вывод Python logging / БД.")
                    font.pixelSize: 11
                    color: Material.color(Material.Grey, Material.Shade400)
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }

        // Сводка
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Rectangle {
                Layout.fillWidth: true
                radius: 10
                color: Qt.rgba(0.9, 0.25, 0.2, 0.12)
                border.width: 1
                border.color: "#c62828"
                implicitHeight: 40

                Label {
                    anchors.centerIn: parent
                    text: statNonce >= 0 ? ("ERR " + countLevel("ERR")) : ""
                    font.bold: true
                    font.pixelSize: 13
                    color: "#ef9a9a"
                }
            }

            Rectangle {
                Layout.fillWidth: true
                radius: 10
                color: Qt.rgba(1, 0.6, 0, 0.1)
                border.width: 1
                border.color: "#F57C00"
                implicitHeight: 40

                Label {
                    anchors.centerIn: parent
                    text: statNonce >= 0 ? ("WARN " + countLevel("WARN")) : ""
                    font.bold: true
                    font.pixelSize: 13
                    color: "#FFCC80"
                }
            }

            Rectangle {
                Layout.fillWidth: true
                radius: 10
                color: Qt.rgba(0.1, 0.55, 0.55, 0.15)
                border.width: 1
                border.color: Material.accent
                implicitHeight: 40

                Label {
                    anchors.centerIn: parent
                    text: statNonce >= 0 ? ("INFO " + countLevel("INFO")) : ""
                    font.bold: true
                    font.pixelSize: 13
                    color: Material.accent
                }
            }
        }

        TextField {
            id: searchField
            Layout.fillWidth: true
            placeholderText: qsTr("Поиск по тексту…")
            selectByMouse: true
            onTextChanged: applyFilter()
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Button {
                text: qsTr("Все")
                flat: filterTag !== "ALL"
                highlighted: filterTag === "ALL"
                onClicked: {
                    filterTag = "ALL"
                    applyFilter()
                }
            }

            Button {
                text: qsTr("Ошибки")
                flat: filterTag !== "ERR"
                highlighted: filterTag === "ERR"
                onClicked: {
                    filterTag = "ERR"
                    applyFilter()
                }
            }

            Button {
                text: qsTr("Предупреждения")
                flat: filterTag !== "WARN"
                highlighted: filterTag === "WARN"
                onClicked: {
                    filterTag = "WARN"
                    applyFilter()
                }
            }

            Button {
                text: qsTr("Инфо")
                flat: filterTag !== "INFO"
                highlighted: filterTag === "INFO"
                onClicked: {
                    filterTag = "INFO"
                    applyFilter()
                }
            }

            Item {
                Layout.fillWidth: true
            }

            ToolButton {
                text: "＋"
                font.pixelSize: 18
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Добавить тестовую запись INFO")
                onClicked: appendLiveLine("INFO", qsTr("Тест: проверка журнала ") + Qt.formatTime(new Date(), "hh:mm:ss"))
            }

            ToolButton {
                text: "⌫"
                font.pixelSize: 16
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Очистить отфильтрованный список (полный журнал сохраняется до сброса демо)")
                onClicked: {
                    for (var i = fullLogModel.count - 1; i >= 0; i--) {
                        var row = fullLogModel.get(i)
                        if (filterTag !== "ALL" && row.level !== filterTag)
                            continue
                        var q = searchField.text.toLowerCase().trim()
                        if (q.length > 0 && String(row.text).toLowerCase().indexOf(q) < 0)
                            continue
                        fullLogModel.remove(i)
                    }
                    statNonce++
                    applyFilter()
                }
            }

            ToolButton {
                text: "↻"
                font.pixelSize: 16
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Сбросить демо-данные")
                onClicked: {
                    searchField.text = ""
                    filterTag = "ALL"
                    seedDemo()
                }
            }
        }

        Label {
            visible: displayModel.count === 0
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.WordWrap
            text: qsTr("Нет записей по текущему фильтру или поиску.")
            font.pixelSize: 13
            color: Material.color(Material.Grey, Material.Shade500)
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 10
            model: displayModel

            delegate: Rectangle {
                width: ListView.view.width
                height: innerCol.height + 20
                radius: 12
                color: cardBg
                border.width: 1
                border.color: cardBorder

                Rectangle {
                    width: 4
                    height: parent.height - 8
                    anchors.left: parent.left
                    anchors.leftMargin: 6
                    anchors.verticalCenter: parent.verticalCenter
                    radius: 2
                    color: model.level === "ERR" ? "#e53935" : (model.level === "WARN" ? "#FB8C00" : Material.accent)
                }

                ColumnLayout {
                    id: innerCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 10
                    anchors.leftMargin: 18
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Rectangle {
                            radius: 6
                            color: model.level === "ERR" ? Qt.rgba(0.9, 0.2, 0.2, 0.25) : (model.level === "WARN" ? Qt.rgba(1, 0.55, 0, 0.2) : Qt.rgba(0, 0.75, 0.8, 0.2))
                            implicitWidth: lvlLabel.width + 14
                            implicitHeight: 24

                            Label {
                                id: lvlLabel
                                anchors.centerIn: parent
                                text: model.level
                                font.bold: true
                                font.pixelSize: 10
                                color: model.level === "ERR" ? "#ffcdd2" : (model.level === "WARN" ? "#ffe0b2" : "#b2ebf2")
                            }
                        }

                        Item {
                            Layout.fillWidth: true
                        }

                        Label {
                            text: model.time
                            font.pixelSize: 11
                            font.family: "monospace"
                            color: Material.color(Material.Grey, Material.Shade500)
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        text: model.text
                        wrapMode: Text.WordWrap
                        font.pixelSize: 13
                        color: Material.foreground
                        lineHeight: 1.15
                    }
                }
            }
        }
    }
}
