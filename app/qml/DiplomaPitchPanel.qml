import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Вкладка промежуточного питча: список 12 экранов + StackView, общая модель жестов.
 */
Item {
    id: diplomaPitchRoot

    function handleBack(stack) {
        if (!stack)
            return
        if (stack.depth > 1) {
            stack.pop()
            return
        }
        stack.replace(stack.currentItem, Qt.resolvedUrl("PitchOverview.qml"), {
                          "gestStack": stack,
                          "pitchHost": diplomaPitchRoot
                      })
    }

    function bumpSamples(gestureName, recorded) {
        if (!gestureName || recorded <= 0)
            return
        for (var i = 0; i < pitchGestureModel.count; i++) {
            var row = pitchGestureModel.get(i)
            if (row.name === gestureName) {
                var s = Math.min(999, row.samples + recorded)
                pitchGestureModel.setProperty(i, "samples", s)
                var ready = s >= 20 ? qsTr("Готово к обучению") : (qsTr("Накопление: ") + s + "/20")
                pitchGestureModel.setProperty(i, "readyStr", ready)
                return
            }
        }
    }

    function openScreen(index) {
        while (pitchStack.depth > 0)
            pitchStack.pop(StackView.Immediate)
        var urls = [
            "HomeScreen.qml",
            "GestureRecognitionScreen.qml",
            "AddGestureScreen.qml",
            "RecordingExamplesScreen.qml",
            "GestureListScreen.qml",
            "GestureCommandBindScreen.qml",
            "ModelTrainingScreen.qml",
            "ModelCompareScreen.qml",
            "MessagesLogScreen.qml",
            "SystemSettingsScreen.qml",
            "HelpScreen.qml",
            "GestureTestScreen.qml"
        ]
        var url = Qt.resolvedUrl(urls[index])
        pitchStack.push(url, {
                             "gestStack": pitchStack,
                             "gestureCatalog": pitchGestureModel,
                             "navRoot": diplomaPitchRoot,
                             "pitchHost": diplomaPitchRoot
                         })
        screenListView.currentIndex = index
    }

    ListModel {
        id: pitchGestureModel
        ListElement {
            name: "Свайп вправо"
            samples: 24
            readyStr: "Готово к обучению"
        }
        ListElement {
            name: "Большой палец вверх"
            samples: 20
            readyStr: "Готово к обучению"
        }
        ListElement {
            name: "Ладонь (стоп)"
            samples: 7
            readyStr: "Накопление: 7/20"
        }
    }

    ListModel {
        id: pitchScreenList
        ListElement {
            title: "1. Главный экран"
        }
        ListElement {
            title: "2. Распознавание жестов"
        }
        ListElement {
            title: "3. Новый жест"
        }
        ListElement {
            title: "4. Запись примеров"
        }
        ListElement {
            title: "5. Список жестов"
        }
        ListElement {
            title: "6. Привязка к командам"
        }
        ListElement {
            title: "7. Обучение модели"
        }
        ListElement {
            title: "8. Сравнение моделей"
        }
        ListElement {
            title: "9. Сообщения и ошибки"
        }
        ListElement {
            title: "10. Настройки"
        }
        ListElement {
            title: "11. Справка"
        }
        ListElement {
            title: "12. Тестирование"
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.preferredWidth: 230
            Layout.fillHeight: true
            color: "#1e1e2e"
            border.width: 1
            border.color: "#333"

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    color: Material.accent

                    Label {
                        anchors.centerIn: parent
                        text: qsTr("Питч диплома")
                        font.pixelSize: 15
                        font.bold: true
                        color: "white"
                    }
                }

                Label {
                    Layout.fillWidth: true
                    Layout.leftMargin: 12
                    Layout.rightMargin: 12
                    Layout.topMargin: 8
                    text: qsTr("12 экранов — клик для показа")
                    font.pixelSize: 11
                    wrapMode: Text.WordWrap
                    color: Material.color(Material.Grey, Material.Shade500)
                }

                ListView {
                    id: screenListView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: pitchScreenList
                    boundsBehavior: Flickable.StopAtBounds

                    delegate: ItemDelegate {
                        width: screenListView.width
                        text: model.title
                        highlighted: ListView.isCurrentItem
                        onClicked: diplomaPitchRoot.openScreen(index)
                    }
                }
            }
        }

        StackView {
            id: pitchStack
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            Component.onCompleted: openScreen(0)
        }
    }
}
