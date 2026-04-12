import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 11. Справка.
 */
Item {
    id: root

    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        RowLayout {
            Layout.fillWidth: true

            ToolButton {
                text: "←"
                font.pixelSize: 18
                onClicked: (pitchHost && pitchHost.handleBack) ? pitchHost.handleBack(gestStack) : gestStack.pop()
            }

            Label {
                text: qsTr("Справка (11)")
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
            contentHeight: helpText.height + 24
            clip: true

            Label {
                id: helpText
                width: parent.width
                wrapMode: Text.WordWrap
                color: Material.foreground
                font.pixelSize: 14
                text: qsTr("Краткая инструкция\n\n"
                            + "1) На главном экране модуля «Жесты» включите камеру при необходимости.\n"
                            + "2) Создайте класс жеста, запишите не менее 20 примеров с разным наклоном руки.\n"
                            + "3) Привяжите жест к команде, обучите или сравните модели, затем включите распознавание.\n\n"
                            + "Рекомендации по записи жестов\n\n"
                            + "• Делайте паузу между образцами, меняйте лёгкий наклон и расстояние до камеры.\n"
                            + "• Не убирайте руку резко из кадра до окончания записи образца.\n\n"
                            + "Освещение и фон\n\n"
                            + "• Равномерный свет без жёсткой тени на кисти; избегайте окна сзади.\n"
                            + "• Контрастный фон без движущихся объектов позади руки.\n\n"
                            + "Основные режимы\n\n"
                            + "• Распознавание — поток с камеры и классификация жестов.\n"
                            + "• Обучение — сбор примеров и переобучение модели.\n"
                            + "• Тест — проверка жеста без постоянного режима распознавания.")
            }
        }
    }
}
