import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 8. Сравнение моделей — карточки, шкалы метрик, вывод и пояснения.
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
        spacing: 8

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
                    text: qsTr("Сравнение моделей")
                    font.pixelSize: 20
                    font.bold: true
                    color: Material.foreground
                }

                Label {
                    text: qsTr("KNN и SVM на одной выборке (демо-значения до подключения офлайн-метрик из cv/)")
                    font.pixelSize: 12
                    color: Material.color(Material.Grey, Material.Shade400)
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }

        ScrollView {
            id: cmpScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            ColumnLayout {
                width: cmpScroll.availableWidth
                spacing: 14

                // Краткие профили моделей
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 118
                        radius: 14
                        color: cardColor
                        border.width: 1
                        border.color: Material.accent

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 6

                            Label {
                                text: "KNN (k=5)"
                                font.bold: true
                                font.pixelSize: 16
                                color: Material.accent
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                font.pixelSize: 12
                                color: Material.color(Material.Grey, Material.Shade300)
                                text: qsTr("Быстрый старт, мало гиперпараметров. Хорош при небольшом числе классов и похожей геометрии руки.")
                            }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 118
                        radius: 14
                        color: cardColor
                        border.width: 1
                        border.color: "#7C4DFF"

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 6

                            Label {
                                text: "SVM (RBF)"
                                font.bold: true
                                font.pixelSize: 16
                                color: "#B388FF"
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                font.pixelSize: 12
                                color: Material.color(Material.Grey, Material.Shade300)
                                text: qsTr("Часто даёт чуть выше обобщение на тесте. Дольше обучение, чувствительнее к масштабу признаков.")
                            }
                        }
                    }
                }

                // Таблица метрик + шкалы
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: cardColor
                    border.width: 1
                    border.color: cardBorder
                    implicitHeight: metricsInner.implicitHeight + 24

                    ColumnLayout {
                        id: metricsInner
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 14

                        Label {
                            text: qsTr("Метрики классификации (0–1)")
                            font.bold: true
                            font.pixelSize: 14
                            color: Material.foreground
                        }

                        CompareMetricRow {
                            Layout.fillWidth: true
                            name: "Accuracy"
                            knnVal: 0.91
                            svmVal: 0.94
                            knnHint: qsTr("Доля верных предсказаний")
                        }

                        CompareMetricRow {
                            Layout.fillWidth: true
                            name: "Precision"
                            knnVal: 0.88
                            svmVal: 0.92
                            knnHint: qsTr("Точность среди срабатываний класса")
                        }

                        CompareMetricRow {
                            Layout.fillWidth: true
                            name: "Recall"
                            knnVal: 0.90
                            svmVal: 0.93
                            knnHint: qsTr("Полнота — как долю найденных истинных жестов")
                        }

                        CompareMetricRow {
                            Layout.fillWidth: true
                            name: "F1-score"
                            knnVal: 0.89
                            svmVal: 0.925
                            knnHint: qsTr("Гармоническое среднее precision и recall")
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: cardBorder
                        }

                        Label {
                            text: qsTr("Затраты времени")
                            font.bold: true
                            font.pixelSize: 14
                            color: Material.foreground
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 16

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 4

                                Label {
                                    text: qsTr("Обучение")
                                    font.pixelSize: 11
                                    color: Material.color(Material.Grey, Material.Shade400)
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Label {
                                        text: "KNN"
                                        font.pixelSize: 12
                                        color: Material.accent
                                        Layout.preferredWidth: 36
                                    }

                                    Label {
                                        text: "0,4 с"
                                        font.pixelSize: 13
                                        color: Material.foreground
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Label {
                                        text: "SVM"
                                        font.pixelSize: 12
                                        color: "#B388FF"
                                        Layout.preferredWidth: 36
                                    }

                                    Label {
                                        text: "2,1 с"
                                        font.pixelSize: 13
                                        color: Material.foreground
                                    }
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 4

                                Label {
                                    text: qsTr("Инференс (оценка)")
                                    font.pixelSize: 11
                                    color: Material.color(Material.Grey, Material.Shade400)
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Label {
                                        text: "KNN"
                                        font.pixelSize: 12
                                        color: Material.accent
                                        Layout.preferredWidth: 36
                                    }

                                    Label {
                                        text: "~2 ms / кадр"
                                        font.pixelSize: 13
                                        color: Material.foreground
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Label {
                                        text: "SVM"
                                        font.pixelSize: 12
                                        color: "#B388FF"
                                        Layout.preferredWidth: 36
                                    }

                                    Label {
                                        text: "~5 ms / кадр"
                                        font.pixelSize: 13
                                        color: Material.foreground
                                    }
                                }
                            }
                        }
                    }
                }

                // Итог
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: Qt.rgba(0.15, 0.35, 0.42, 0.35)
                    border.width: 1
                    border.color: Qt.rgba(0.2, 0.75, 0.85, 0.5)
                    implicitHeight: verdictCol.implicitHeight + 24

                    ColumnLayout {
                        id: verdictCol
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 8

                        Label {
                            text: qsTr("Вывод для демо-данных")
                            font.bold: true
                            font.pixelSize: 15
                            color: Material.accent
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            font.pixelSize: 13
                            color: Material.foreground
                            text: qsTr("SVM немного выигрывает по качеству (F1, precision), KNN — по скорости обучения и инференса. Для прототипа и малых наборов жестов удобнее KNN; если важна последняя доля точности на тесте — имеет смысл сравнить обе модели на ваших реальных логах после обучения в cv/.")
                        }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    font.pixelSize: 11
                    color: Material.color(Material.Grey, Material.Shade500)
                    text: qsTr("Числа выше — иллюстрация интерфейса. Подставьте свои accuracy / F1 из отчёта обучения (например, sklearn classification_report) для честного сравнения.")
                }

                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 16
                }
            }
        }
    }

    /** Одна строка метрики: название + две шкалы KNN / SVM */
    component CompareMetricRow: ColumnLayout {
        property string name: ""
        property real knnVal: 0
        property real svmVal: 0
        property string knnHint: ""

        spacing: 6

        RowLayout {
            Layout.fillWidth: true

            Label {
                text: name
                Layout.preferredWidth: 88
                font.pixelSize: 13
                font.bold: true
                color: Material.foreground
            }

            Label {
                text: Math.round(knnVal * 100) + "%"
                Layout.preferredWidth: 40
                horizontalAlignment: Text.AlignRight
                font.pixelSize: 12
                color: Material.accent
            }

            ProgressBar {
                Layout.fillWidth: true
                from: 0
                to: 1
                value: knnVal
                Material.accent: Material.Cyan
            }
        }

        RowLayout {
            Layout.fillWidth: true

            Item {
                Layout.preferredWidth: 88
            }

            Label {
                text: Math.round(svmVal * 100) + "%"
                Layout.preferredWidth: 40
                horizontalAlignment: Text.AlignRight
                font.pixelSize: 12
                color: "#B388FF"
            }

            ProgressBar {
                Layout.fillWidth: true
                from: 0
                to: 1
                value: svmVal
                Material.accent: "#7C4DFF"
            }
        }

        Label {
            visible: knnHint.length > 0
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            font.pixelSize: 10
            color: Material.color(Material.Grey, Material.Shade500)
            text: knnHint
        }
    }
}
