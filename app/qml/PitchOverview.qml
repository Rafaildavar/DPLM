import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material

/**
 * Заглушка стека питча: «выберите экран слева» после «Назад» с корневого экрана.
 */
Item {
    required property StackView gestStack
    property var pitchHost
    property var gestureCatalog
    property var navRoot

    Label {
        anchors.centerIn: parent
        width: parent.width - 40
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.WordWrap
        text: qsTr("Выберите любой из 12 экранов в списке слева.\n\nКамера и демо-данные работают так же, как во вкладке «Жесты».")
        font.pixelSize: 15
        color: Material.color(Material.Grey, Material.Shade400)
    }
}
