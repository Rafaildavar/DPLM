import QtQuick

Canvas {
    id: root

    property string landmarksJson: "[]"
    property int sourceWidth: 640
    property int sourceHeight: 480
    property color strokeColor: "#00E5FF"
    property color pointColor: "#00E5FF"
    property real lineWidth: 2.5
    property real pointRadius: 4.0

    readonly property var edges: [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
        [5, 9], [9, 13], [13, 17]
    ]

    function videoRect() {
        const vw = Math.max(1, width)
        const vh = Math.max(1, height)
        const sw = Math.max(1, sourceWidth)
        const sh = Math.max(1, sourceHeight)
        // CameraPreview uses PreserveAspectCrop.
        const scale = Math.max(vw / sw, vh / sh)
        const rw = sw * scale
        const rh = sh * scale
        return {
            x: (vw - rw) / 2,
            y: (vh - rh) / 2,
            w: rw,
            h: rh
        }
    }

    onLandmarksJsonChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()
    onSourceWidthChanged: requestPaint()
    onSourceHeightChanged: requestPaint()

    onPaint: {
        const ctx = getContext("2d")
        ctx.reset()

        let pts = []
        try {
            pts = JSON.parse(landmarksJson || "[]")
        } catch (e) {
            pts = []
        }
        if (!pts.length)
            return

        const vr = videoRect()
        function px(p) { return vr.x + p[0] * vr.w }
        function py(p) { return vr.y + p[1] * vr.h }

        ctx.strokeStyle = strokeColor
        ctx.lineWidth = lineWidth
        ctx.beginPath()
        for (let i = 0; i < edges.length; i++) {
            const a = edges[i][0]
            const b = edges[i][1]
            if (a < pts.length && b < pts.length) {
                ctx.moveTo(px(pts[a]), py(pts[a]))
                ctx.lineTo(px(pts[b]), py(pts[b]))
            }
        }
        ctx.stroke()

        ctx.fillStyle = pointColor
        for (let j = 0; j < pts.length; j++) {
            ctx.beginPath()
            ctx.arc(px(pts[j]), py(pts[j]), pointRadius, 0, 6.28)
            ctx.fill()
        }
    }
}
