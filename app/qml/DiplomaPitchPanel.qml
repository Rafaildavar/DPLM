import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Вкладка промежуточного питча: список 12 экранов + StackView.
 * Боковая колонка — явная ширина (Row), чтобы StackView не перекрывал список.
 */
Item {
    id: diplomaPitchRoot

    readonly property int sideBarWidth: Math.min(260, Math.max(200, Math.floor(width * 0.26)))

    function handleBack(stack) {
        if (!stack)
            return
        if (stack.depth > 1) {
            stack.pop()
            return
        }
        stack.replace(stack.currentItem, Qt.resolvedUrl("PitchOverview.qml"), {
                          "gestStack": stack,
                          "gestureCatalog": pitchGestureModel,
                          "navRoot": diplomaPitchRoot,
                          "pitchHost": diplomaPitchRoot
                      })
    }

    /** Синхронизация подсветки списка с экраном (в т.ч. после gestStack.push с экрана 3). */
    function setPitchScreenIndex(ix) {
        var i = Number(ix)
        if (!isNaN(i) && i >= 0 && i < pitchScreenList.count)
            screenListView.currentIndex = i
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

    function openScreen(ix) {
        var index = Number(ix)
        if (isNaN(index) || index < 0 || index > 11)
            return
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
        var guard = 0
        while (pitchStack.depth > 0 && guard < 24) {
            pitchStack.pop(StackView.Immediate)
            guard++
        }
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

    Row {
        id: pitchRow
        anchors.fill: parent
        spacing: 0

        Rectangle {
            id: pitchSideBar
            width: diplomaPitchRoot.sideBarWidth
            height: parent.height
            color: "#1e1e2e"
            border.width: 1
            border.color: "#333"
            z: 10

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    color: Material.accent

                    Label {
                        anchors.centerIn: parent
                        text: qsTr("Питч")
                        font.pixelSize: 15
                        font.bold: true
                        color: "white"
                    }
                }

                Label {
                    Layout.fillWidth: true
                    Layout.leftMargin: 10
                    Layout.rightMargin: 10
                    Layout.topMargin: 8
                    text: qsTr("12 экранов — нажмите строку")
                    font.pixelSize: 11
                    wrapMode: Text.WordWrap
                    color: Material.color(Material.Grey, Material.Shade500)
                }

                ListView {
                    id: screenListView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 120
                    clip: true
                    model: pitchScreenList
                    boundsBehavior: Flickable.StopAtBounds
                    interactive: true

                    delegate: Rectangle {
                        width: screenListView.width
                        height: 46
                        color: rowMa.pressed ? "#3d3d55" : (ListView.isCurrentItem ? "#353548" : (rowMa.containsMouse ? "#2c2c40" : "transparent"))

                        Label {
                            anchors.left: parent.left
                            anchors.leftMargin: 10
                            anchors.verticalCenter: parent.verticalCenter
                            width: parent.width - 16
                            text: model.title
                            elide: Text.ElideRight
                            color: Material.foreground
                            font.pixelSize: 13
                        }

                        MouseArea {
                            id: rowMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: diplomaPitchRoot.openScreen(index)
                        }
                    }
                }
            }
        }

        StackView {
            id: pitchStack
            width: pitchRow.width - pitchSideBar.width
            height: pitchRow.height
            clip: true
            z: 1

            Component.onCompleted: openScreen(0)
        }
    }
}
