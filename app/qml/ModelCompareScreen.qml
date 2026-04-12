import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 8. Сравнение моделей (KNN / SVM и метрики).
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            Label {
                text: qsTr("Сравнение моделей (8)")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
                color: Material.foreground
            }
        }

        Label {
            text: qsTr("Демонстрационные значения после кросс-валидации (заглушка).")
            font.pixelSize: 11
            color: Material.color(Material.Grey, Material.Shade500)
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 3
            columnSpacing: 8
            rowSpacing: 6

            Label {
                text: ""
            }
            Label {
                text: "KNN"
                font.bold: true
                color: Material.accent
            }
            Label {
                text: "SVM"
                font.bold: true
                color: Material.accent
            }

            Label {
                text: "Accuracy"
                color: Material.foreground
            }
            Label {
                text: "0.91"
            }
            Label {
                text: "0.94"
            }

            Label {
                text: "Precision"
                color: Material.foreground
            }
            Label {
                text: "0.88"
            }
            Label {
                text: "0.92"
            }

            Label {
                text: "Recall"
                color: Material.foreground
            }
            Label {
                text: "0.90"
            }
            Label {
                text: "0.93"
            }

            Label {
                text: "F1-score"
                color: Material.foreground
            }
            Label {
                text: "0.89"
            }
            Label {
                text: "0.925"
            }

            Label {
                text: qsTr("Время обучения")
                color: Material.foreground
            }
            Label {
                text: "0.4 c"
            }
            Label {
                text: "2.1 c"
            }

            Label {
                text: qsTr("Время инференса")
                color: Material.foreground
            }
            Label {
                text: "~2 ms"
            }
            Label {
                text: "~5 ms"
            }
        }

        Item {
            Layout.fillHeight: true
        }
    }
}
