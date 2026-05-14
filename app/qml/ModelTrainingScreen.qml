import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 7. Обучение модели — сводка данных, выбор алгоритма, прогресс, отчёт.
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    property string trainResult: ""
    property bool trainingBusy: false
    property real trainingProgress: 0

    readonly property color cardBg: "#232336"
    readonly property color cardBorder: "#3d3d55"

    StackView.onStatusChanged: {
        if (StackView.status === StackView.Active && pitchHost && pitchHost.setPitchScreenIndex)
            pitchHost.setPitchScreenIndex(6)
    }

    function totalSamplesInCatalog() {
        if (!gestureCatalog)
            return 0
        var t = 0
        for (var i = 0; i < gestureCatalog.count; i++)
            t += Number(gestureCatalog.get(i).samples) || 0
        return t
    }

    function classesReadyCount() {
        if (!gestureCatalog)
            return 0
        var n = 0
        for (var i = 0; i < gestureCatalog.count; i++) {
            if (Number(gestureCatalog.get(i).samples) >= 20)
                n++
        }
        return n
    }

    function algoDescription() {
        var i = algoPick.currentIndex
        if (i === 0)
            return qsTr("KNeighborsClassifier: быстро, мало гиперпараметров, хорошая база для прототипа.")
        if (i === 1)
            return qsTr("SVC с RBF-ядром: часто выше качество на тесте, дольше обучение.")
        return qsTr("RandomForestClassifier — заглушка в UI; подключение в cv/ по необходимости.")
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 0

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
                    text: qsTr("Обучение модели")
                    font.pixelSize: 20
                    font.bold: true
                    color: Material.foreground
                }

                Label {
                    text: qsTr("Подготовка классификатора по признакам MediaPipe (42D) — демо-поток с отчётом.")
                    font.pixelSize: 11
                    color: Material.color(Material.Grey, Material.Shade400)
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }

        ScrollView {
            id: scr
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.topMargin: 12
            clip: true

            ColumnLayout {
                width: scr.availableWidth
                spacing: 14

                // Сводка по данным
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardBg
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: dsCol.implicitHeight + 24

                    RowLayout {
                        id: dsCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 14

                        Rectangle {
                            Layout.preferredWidth: 52
                            Layout.preferredHeight: 52
                            radius: 12
                            color: Qt.rgba(0, 0.75, 0.85, 0.18)

                            Label {
                                anchors.centerIn: parent
                                text: "∑"
                                font.pixelSize: 26
                                color: Material.accent
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: qsTr("Обучающая выборка")
                                font.bold: true
                                font.pixelSize: 14
                                color: Material.accent
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                font.pixelSize: 13
                                color: Material.foreground
                                text: gestureCatalog
                                      ? qsTr("Классов в каталоге: %1 · всего примеров: %2 · готово к обучению (≥20 шт.): %3")
                                      .arg(gestureCatalog.count).arg(totalSamplesInCatalog()).arg(classesReadyCount())
                                      : qsTr("Каталог жестов недоступен на этом экране — числа ниже ориентировочные.")
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                font.pixelSize: 11
                                color: Material.color(Material.Grey, Material.Shade500)
                                text: qsTr("Реальное обучение: скрипты в cv/ и сохранение в models/knn.pkl + models/classes.json.")
                            }
                        }
                    }
                }

                // Алгоритм
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardBg
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: algoCol.implicitHeight + 24

                    ColumnLayout {
                        id: algoCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10

                        Label {
                            text: qsTr("Алгоритм")
                            font.bold: true
                            font.pixelSize: 14
                            color: Material.accent
                        }

                        ComboBox {
                            id: algoPick
                            Layout.fillWidth: true
                            model: [qsTr("KNN (k=5)"), qsTr("SVM (RBF kernel)"), qsTr("Random Forest (заглушка)")]
                            onCurrentIndexChanged: {}
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            font.pixelSize: 12
                            color: Material.color(Material.Grey, Material.Shade400)
                            text: algoDescription()
                        }
                    }
                }

                // Гиперпараметры (визуальные якоря для «профессионального» вида)
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardBg
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: hpCol.implicitHeight + 24

                    ColumnLayout {
                        id: hpCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10

                        Label {
                            text: qsTr("Параметры (демо)")
                            font.bold: true
                            font.pixelSize: 14
                            color: Material.accent
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 16

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 4

                                Label {
                                    text: qsTr("Валидация")
                                    font.pixelSize: 11
                                    color: Material.color(Material.Grey, Material.Shade500)
                                }

                                ComboBox {
                                    id: cvPick
                                    Layout.fillWidth: true
                                    model: ["5-fold CV", "10-fold CV", "Hold-out 20%"]
                                    currentIndex: 0
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 4

                                Label {
                                    text: qsTr("Сид")
                                    font.pixelSize: 11
                                    color: Material.color(Material.Grey, Material.Shade500)
                                }

                                SpinBox {
                                    id: seedSpin
                                    Layout.fillWidth: true
                                    from: 0
                                    to: 999999
                                    value: 42
                                    editable: true
                                }
                            }
                        }
                    }
                }

                Button {
                    Layout.fillWidth: true
                    Layout.maximumWidth: 480
                    Layout.alignment: Qt.AlignHCenter
                    Layout.preferredHeight: 48
                    highlighted: true
                    enabled: !trainingBusy
                    text: trainingBusy ? qsTr("Идёт обучение…") : qsTr("Запустить обучение")
                    onClicked: {
                        trainResult = ""
                        trainingProgress = 0
                        trainingBusy = true
                        progTick.start()
                        busyTrain.start()
                    }
                }

                // Прогресс
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardBg
                    border.width: 1
                    border.color: cardBorder
                    visible: trainingBusy || trainingProgress > 0
                    implicitHeight: visible ? (progInner.implicitHeight + 24) : 0

                    ColumnLayout {
                        id: progInner
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 8
                        visible: parent.visible

                        RowLayout {
                            Layout.fillWidth: true

                            Label {
                                text: qsTr("Прогресс")
                                font.bold: true
                                font.pixelSize: 13
                                color: Material.foreground
                            }

                            Item {
                                Layout.fillWidth: true
                            }

                            Label {
                                text: Math.round(trainingProgress) + "%"
                                font.family: "monospace"
                                font.pixelSize: 13
                                color: Material.accent
                            }
                        }

                        ProgressBar {
                            Layout.fillWidth: true
                            from: 0
                            to: 100
                            value: trainingProgress
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            visible: trainingBusy

                            BusyIndicator {
                                running: true
                                implicitWidth: 28
                                implicitHeight: 28
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                font.pixelSize: 11
                                color: Material.color(Material.Grey, Material.Shade400)
                                text: qsTr("Загрузка признаков → нормализация → обучение → кросс-валидация (имитация задержки).")
                            }
                        }
                    }
                }

                // Отчёт
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: Qt.rgba(0.12, 0.4, 0.35, 0.25)
                    border.width: 1
                    border.color: Qt.rgba(0.2, 0.85, 0.75, 0.45)
                    visible: trainResult.length > 0
                    implicitHeight: trainResult.length > 0 ? (repCol.implicitHeight + 24) : 0

                    ColumnLayout {
                        id: repCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10
                        visible: trainResult.length > 0

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: "✓"
                                font.pixelSize: 22
                                color: "#69F0AE"
                            }

                            Label {
                                text: qsTr("Отчёт")
                                font.bold: true
                                font.pixelSize: 15
                                color: "#B2DFDB"
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            font.pixelSize: 13
                            color: Material.foreground
                            text: trainResult
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    radius: 10
                    color: "#1e1e2a"
                    border.width: 1
                    border.color: "#333"
                    implicitHeight: footCol.implicitHeight + 20

                    ColumnLayout {
                        id: footCol
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 4

                        Label {
                            text: qsTr("Интеграция")
                            font.bold: true
                            font.pixelSize: 11
                            color: Material.color(Material.Grey, Material.Shade400)
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            font.pixelSize: 11
                            color: Material.color(Material.Grey, Material.Shade500)
                            text: qsTr("Подключите сюда вызов sklearn (fit, classification_report) из Python или существующий train_classifier в cv/. Текущий экран демонстрирует UX питча.")
                        }
                    }
                }

                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 16
                }
            }
        }
    }

    Timer {
        id: progTick
        interval: 90
        repeat: true
        onTriggered: {
            if (!trainingBusy) {
                progTick.stop()
                if (trainingProgress < 100)
                    trainingProgress = 100
                return
            }
            if (trainingProgress < 92)
                trainingProgress += 3 + Math.random() * 5
            else
                trainingProgress += 1
            trainingProgress = Math.min(trainingProgress, 94)
        }
    }

    Timer {
        id: busyTrain
        interval: 2400
        repeat: false
        onTriggered: {
            progTick.stop()
            trainingBusy = false
            trainingProgress = 100
            var algo = algoPick.currentText
            var cv = cvPick.currentText
            trainResult = qsTr("Статус: успешно (демо)\n\n")
                    + qsTr("Алгоритм: %1\n").arg(algo)
                    + qsTr("Валидация: %1\n").arg(cv)
                    + qsTr("random_state: %1\n\n").arg(seedSpin.value)
                    + qsTr("Метрики (имитация): accuracy ≈ 0,91 · macro-F1 ≈ 0,88\n")
                    + qsTr("Выходные файлы (цель): models/knn.pkl, models/classes.json, models/feature_dim.txt\n\n")
                    + qsTr("Следующий шаг: сравните модели на экране «Сравнение моделей» или запустите распознавание.")
        }
    }
}
