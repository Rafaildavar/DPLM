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
    property bool showStatusBanner: true
    /** Рисовать скелет руки (нормализованные точки 0…1) поверх кадра */
    property bool showHandOverlay: false
    property string handLandmarksJson: "[]"
    property string activeStatusText: qsTr("Камера активна — покажите жест и нажмите «Зафиксировать образец» после каждого положения руки.")
    property string inactiveStatusText: qsTr("Камера неактивна. Нажмите «Запустить камеру».")
    readonly property string statusText: appController.isCameraActive ? activeStatusText : inactiveStatusText

    implicitHeight: previewHeight
    color: "#0f1b4d"
    radius: cornerRadius
    border.width: 1
    border.color: "#00cfe8"
    clip: true

    Column {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8

        Rectangle {
            width: parent.width
            height: showStatusBanner ? 36 : 0
            visible: showStatusBanner
            radius: 10
            color: appController.isCameraActive ? "#0f8b6c" : "#5a3b3b"
            border.width: 1
            border.color: appController.isCameraActive ? "#48f0bd" : "#c79090"

            Text {
                anchors.fill: parent
                anchors.margins: 10
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                text: statusText
                font.pixelSize: 12
                color: "white"
            }
        }

        Rectangle {
            width: parent.width
            height: parent.height - (showStatusBanner ? 44 : 0)
            radius: Math.max(8, cornerRadius - 4)
            color: "#13215d"
            border.width: 1
            border.color: "#27d7ee"
            clip: true

            Image {
                id: camImage
                anchors.fill: parent
                anchors.margins: 2
                fillMode: showHandOverlay ? Image.PreserveAspectFit : Image.PreserveAspectCrop
                asynchronous: false
                cache: false
                smooth: true
                visible: appController.isCameraActive
                source: appController.isCameraActive
                        ? ("image://dplmcam/preview?t=" + appController.cameraPreviewRevision)
                        : ""
            }

            // Совпадает с «нарисованным» видеокадром (учёт letterbox при AspectFit)
            Canvas {
                id: handCanvas
                z: 2
                visible: showHandOverlay && appController.isCameraActive && paintedW > 0 && paintedH > 0
                opacity: 0.95
                anchors.horizontalCenter: camImage.horizontalCenter
                anchors.verticalCenter: camImage.verticalCenter
                width: paintedW
                height: paintedH

                readonly property real paintedW: camImage.paintedWidth
                readonly property real paintedH: camImage.paintedHeight

                onPaintedWChanged: requestPaint()
                onPaintedHChanged: requestPaint()

                Connections {
                    target: appController
                    function onCameraPreviewRevisionChanged() {
                        if (showHandOverlay)
                            handCanvas.requestPaint()
                    }
                }
                Connections {
                    target: root
                    function onHandLandmarksJsonChanged() {
                        if (showHandOverlay)
                            handCanvas.requestPaint()
                    }
                }

                onPaint: {
                    const ctx = getContext("2d")
                    ctx.reset()
                    let raw = []
                    try {
                        raw = JSON.parse(handLandmarksJson || "[]")
                    } catch (e) {
                        raw = []
                    }
                    if (!raw.length)
                        return

                    // Нормализуем формат: поддержим и старый (одна рука: [[x,y],...]),
                    // и новый (несколько рук: [[[x,y],...], [[x,y],...]]).
                    let hands = []
                    if (raw.length > 0 && raw[0].length === 2 && typeof raw[0][0] === "number") {
                        hands = [raw]
                    } else {
                        hands = raw
                    }

                    const w = width
                    const h = height
                    const edges = [
                        [0, 1], [1, 2], [2, 3], [3, 4],
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        [0, 9], [9, 10], [10, 11], [11, 12],
                        [0, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20],
                        [5, 9], [9, 13], [13, 17]
                    ]
                    const handColors = [
                        { line: Qt.rgba(0, 0.95, 0.85, 0.95), point: "#00E5FF" },
                        { line: Qt.rgba(1.0, 0.55, 0.20, 0.95), point: "#FFB400" }
                    ]

                    for (let hi = 0; hi < hands.length; hi++) {
                        const pts = hands[hi]
                        if (!pts || !pts.length)
                            continue
                        const palette = handColors[hi % handColors.length]
                        ctx.strokeStyle = palette.line
                        ctx.lineWidth = 2.5
                        ctx.beginPath()
                        for (let e = 0; e < edges.length; e++) {
                            const a = edges[e][0]
                            const b = edges[e][1]
                            if (a < pts.length && b < pts.length) {
                                ctx.moveTo(pts[a][0] * w, pts[a][1] * h)
                                ctx.lineTo(pts[b][0] * w, pts[b][1] * h)
                            }
                        }
                        ctx.stroke()
                        ctx.fillStyle = palette.point
                        for (let i = 0; i < pts.length; i++) {
                            ctx.beginPath()
                            ctx.arc(pts[i][0] * w, pts[i][1] * h, 4, 0, 6.28)
                            ctx.fill()
                        }
                    }
                }
                Component.onCompleted: requestPaint()
                onWidthChanged: requestPaint()
                onHeightChanged: requestPaint()
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
    }
}
