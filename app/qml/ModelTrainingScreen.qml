import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 7. Обучение модели.
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    property string trainResult: ""
    property bool trainingBusy: false

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
                text: qsTr("Обучение модели (7)")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
                color: Material.foreground
            }
        }

        Label {
            text: qsTr("Алгоритм классификации")
            font.bold: true
            color: Material.foreground
        }

        ComboBox {
            id: algoPick
            Layout.fillWidth: true
            model: [qsTr("KNN (k=5)"), qsTr("SVM (RBF)"), qsTr("Случайный лес (заглушка)")]
        }

        Label {
            text: qsTr("Объём обучающей выборки (оценка)")
            font.bold: true
            color: Material.foreground
        }

        Label {
            text: qsTr("Всего размеченных примеров по классам: 184 (демо)")
            font.pixelSize: 14
            color: Material.color(Material.Grey, Material.Shade400)
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        Button {
            Layout.fillWidth: true
            text: trainingBusy ? qsTr("Обучение…") : qsTr("Запустить обучение")
            enabled: !trainingBusy
            highlighted: true
            onClicked: {
                trainResult = ""
                trainingBusy = true
                busyTrain.start()
            }
        }

        BusyIndicator {
            Layout.alignment: Qt.AlignHCenter
            running: trainingBusy
            visible: trainingBusy
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 120
            radius: 12
            color: "#2a2a3e"
            visible: trainResult.length > 0

            ScrollView {
                anchors.fill: parent
                anchors.margins: 10
                Label {
                    width: parent.width
                    text: trainResult
                    wrapMode: Text.WordWrap
                    color: Material.foreground
                }
            }
        }

        Item {
            Layout.fillHeight: true
        }
    }

    Timer {
        id: busyTrain
        interval: 2200
        repeat: false
        onTriggered: {
            trainingBusy = false
            trainResult = qsTr("Готово.\nАлгоритм: ") + algoPick.currentText + qsTr("\nДемо: веса сохранены в каталог модели (интеграция с train_classifier — TODO).")
        }
    }
}
