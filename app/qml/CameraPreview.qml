import QtQuick
import QtQuick.Controls.Material

/**
 * Живое превью с камеры через appController + image://dplmcam/preview
 * (OpenCV в Python, те же параметры, что cv/realtime_infer.py).
 */
Rectangle {
    id: root

    property int previewHeight: 200
    property int cornerRadius: 16

    implicitHeight: previewHeight
    color: "#1E1E2E"
    radius: cornerRadius
    border.width: 2
    border.color: appController.isCameraActive ? Material.accent : Material.color(Material.Grey, Material.Shade700)
    clip: true

    Image {
        id: camImage
        anchors.fill: parent
        anchors.margins: 2
        fillMode: Image.PreserveAspectCrop
        asynchronous: false
        cache: false
        smooth: true
        visible: appController.isCameraActive
        source: appController.isCameraActive
                ? ("image://dplmcam/preview?t=" + appController.cameraPreviewRevision)
                : ""
    }

    Column {
        anchors.centerIn: parent
        spacing: 8
        visible: !appController.isCameraActive
        width: parent.width - 24

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: "◼"
            font.pixelSize: 36
            color: Material.color(Material.Grey, Material.Shade500)
        }

        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.WordWrap
            text: qsTr("Нажмите «Запустить камеру» — поток идёт через OpenCV (как в realtime_infer.py).")
            font.pixelSize: 12
            color: Material.color(Material.Grey, Material.Shade400)
        }
    }
}
