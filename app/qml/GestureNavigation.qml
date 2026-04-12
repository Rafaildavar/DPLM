import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material

/**
 * Навигация по 12 экранам модуля жестов (StackView + каталог жестов).
 */
Item {
    id: gestureNavRoot

    /**
     * Увеличить счётчик примеров для класса с именем gestureName.
     */
    function bumpSamples(gestureName, recorded) {
        if (!gestureName || recorded <= 0)
            return
        for (var i = 0; i < savedGesturesModel.count; i++) {
            var row = savedGesturesModel.get(i)
            if (row.name === gestureName) {
                var s = Math.min(999, row.samples + recorded)
                savedGesturesModel.setProperty(i, "samples", s)
                var ready = s >= 20 ? qsTr("Готово к обучению") : (qsTr("Накопление: ") + s + "/20")
                savedGesturesModel.setProperty(i, "readyStr", ready)
                return
            }
        }
    }

    ListModel {
        id: savedGesturesModel
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

    StackView {
        id: gestureStack
        anchors.fill: parent

        initialItem: HomeScreen {
            gestStack: gestureStack
            gestureCatalog: savedGesturesModel
            navRoot: gestureNavRoot
        }
    }
}
