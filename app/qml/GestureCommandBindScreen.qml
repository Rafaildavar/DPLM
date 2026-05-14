import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Экран 6. Привязка жестов к командам.
 *
 * Реализует правила из docs/BINDING_RULES.md:
 *   R1 — 1:1 жест↔команда (с переспросом при перезаписи)
 *   R3 — фильтр жестов: только is_active=True и model_class_id IS NOT NULL
 *   R6 — предупреждение для опасных действий, не привязанных к двуручному жесту
 *   R7 — валидация action_spec (на сервере: appController.validateActionSpec)
 *   R8 — уникальность имени команды (на сервере: appController.validateCommandName)
 */
Item {
    id: root

    required property StackView gestStack
    property var gestureCatalog
    property var navRoot
    property var pitchHost

    /** Источник жестов из БД (R3). Заполняется при показе экрана. */
    ListModel { id: dbGestureModel }

    /** Категории команд (см. ACTION_SPEC_SCHEMA). Заполняется один раз. */
    ListModel { id: categoryModel }

    /** Действия внутри выбранной категории. Перестраивается при смене. */
    ListModel { id: actionModel }

    property string lastError: ""
    property string lastInfo: ""
    property bool hasOverwriteCandidate: false
    property string overwriteCandidate: ""

    function refreshDbGestures() {
        dbGestureModel.clear()
        var rows = appController.getDbGestures()
        for (var i = 0; i < rows.length; i++) {
            var r = rows[i]
            var displayLabel = r.label
            if (r.boundCommandName && r.boundCommandName.length > 0)
                displayLabel += "  →  " + r.boundCommandName
            dbGestureModel.append({
                "label": String(r.label || ""),
                "displayLabel": String(displayLabel),
                "description": String(r.description || ""),
                "isTwoHands": Boolean(r.isTwoHands || false),
                "boundCommandName": String(r.boundCommandName || "")
            })
        }
    }

    function refreshCategories() {
        categoryModel.clear()
        var cats = appController.getActionCategories()
        for (var i = 0; i < cats.length; i++)
            categoryModel.append({"id": cats[i].id, "label": cats[i].label})
    }

    function refreshActions(categoryId) {
        actionModel.clear()
        if (!categoryId)
            return
        var acts = appController.getActionsForCategory(categoryId)
        for (var i = 0; i < acts.length; i++) {
            var a = acts[i]
            actionModel.append({
                "action": String(a.action),
                "example": JSON.stringify(a.example || {}),
                "fields": JSON.stringify(a.fields || []),
                "fieldHints": JSON.stringify(a.fieldHints || {})
            })
        }
    }

    /** Собрать action_spec на основе выбранного действия и значений полей. */
    function buildActionSpec() {
        if (actionPick.currentIndex < 0)
            return null
        var row = actionModel.get(actionPick.currentIndex)
        var spec = { "action": row.action, "platform": "macos" }
        var raw = paramsField.text.trim()
        if (raw.length > 0) {
            try {
                var extra = JSON.parse(raw)
                if (extra && typeof extra === "object")
                    for (var k in extra)
                        spec[k] = extra[k]
            } catch (e) {
                lastError = qsTr("Параметры: некорректный JSON: ") + e
                return null
            }
        }
        return spec
    }

    /** Текущий выбранный жест: запись из dbGestureModel или null. */
    function currentGesture() {
        if (gesturePick.currentIndex < 0 || dbGestureModel.count === 0)
            return null
        return dbGestureModel.get(gesturePick.currentIndex)
    }

    /** R6: показать ли предупреждение о двуручном жесте. */
    function shouldWarnTwoHands() {
        var g = currentGesture()
        if (!g)
            return false
        if (g.isTwoHands)
            return false
        if (actionPick.currentIndex < 0)
            return false
        var row = actionModel.get(actionPick.currentIndex)
        return appController.isActionDangerous(row.action)
    }

    /** R1: проверить, есть ли уже команда на этом жесте, и нужно ли переспросить. */
    function refreshOverwriteHint() {
        var g = currentGesture()
        if (!g) {
            hasOverwriteCandidate = false
            overwriteCandidate = ""
            return
        }
        var bound = String(g.boundCommandName || "")
        if (bound.length > 0 && bound !== nameField.text.trim()) {
            hasOverwriteCandidate = true
            overwriteCandidate = bound
        } else {
            hasOverwriteCandidate = false
            overwriteCandidate = ""
        }
    }

    Component.onCompleted: {
        refreshCategories()
        refreshDbGestures()
        if (categoryModel.count > 0) {
            categoryPick.currentIndex = 0
            refreshActions(categoryModel.get(0).id)
        }
    }

    StackView.onActivated: refreshDbGestures()

    ScrollView {
        anchors.fill: parent
        clip: true

        ColumnLayout {
            width: root.width
            spacing: 12

            RowLayout {
                Layout.fillWidth: true

                ToolButton {
                    text: "←"
                    font.pixelSize: 18
                    onClicked: (pitchHost && pitchHost.handleBack)
                                ? pitchHost.handleBack(gestStack)
                                : gestStack.pop()
                }

                Label {
                    text: qsTr("Привязка жестов к командам (6)")
                    font.pixelSize: 18
                    font.bold: true
                    Layout.fillWidth: true
                    color: Material.foreground
                }

                ToolButton {
                    text: "↻"
                    ToolTip.text: qsTr("Обновить список жестов из БД")
                    ToolTip.visible: hovered
                    onClicked: root.refreshDbGestures()
                }
            }

            // --- Жест (R3) ---
            Label {
                text: qsTr("Жест из словаря (только активные и обученные)")
                font.bold: true
                color: Material.foreground
            }

            ComboBox {
                id: gesturePick
                Layout.fillWidth: true
                model: dbGestureModel
                textRole: "displayLabel"
                enabled: dbGestureModel.count > 0
                onCurrentIndexChanged: root.refreshOverwriteHint()
            }

            Label {
                visible: dbGestureModel.count === 0
                Layout.fillWidth: true
                text: qsTr("Нет доступных жестов. Запишите примеры и обучите модель — экраны 4 и 7.")
                color: Material.color(Material.Orange)
                wrapMode: Text.WordWrap
            }

            // --- Категория и действие ---
            Label {
                text: qsTr("Категория команды")
                font.bold: true
                color: Material.foreground
            }

            ComboBox {
                id: categoryPick
                Layout.fillWidth: true
                model: categoryModel
                textRole: "label"
                onCurrentIndexChanged: {
                    if (currentIndex >= 0 && categoryModel.count > 0)
                        root.refreshActions(categoryModel.get(currentIndex).id)
                }
            }

            Label {
                text: qsTr("Действие")
                font.bold: true
                color: Material.foreground
            }

            ComboBox {
                id: actionPick
                Layout.fillWidth: true
                model: actionModel
                textRole: "action"
                enabled: actionModel.count > 0
                onCurrentIndexChanged: {
                    if (currentIndex >= 0) {
                        var row = actionModel.get(currentIndex)
                        try {
                            var ex = JSON.parse(row.example || "{}")
                            var copy = {}
                            for (var k in ex)
                                if (k !== "action" && k !== "platform")
                                    copy[k] = ex[k]
                            paramsField.text = JSON.stringify(copy)
                        } catch (e) {
                            paramsField.text = "{}"
                        }
                        if (nameField.text.trim().length === 0)
                            nameField.text = "User: " + row.action
                    }
                }
            }

            Label {
                visible: actionPick.currentIndex >= 0
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                font.pixelSize: 12
                color: Material.color(Material.Grey, Material.Shade400)
                text: {
                    if (actionPick.currentIndex < 0)
                        return ""
                    var row = actionModel.get(actionPick.currentIndex)
                    var hints = {}
                    try { hints = JSON.parse(row.fieldHints || "{}") } catch (e) {}
                    var parts = []
                    for (var k in hints)
                        parts.push("• " + k + ": " + hints[k])
                    return parts.length > 0 ? parts.join("\n") : qsTr("Без дополнительных параметров")
                }
            }

            // --- Параметры (JSON) ---
            Label {
                text: qsTr("Параметры (JSON)")
                font.bold: true
                color: Material.foreground
            }

            TextField {
                id: paramsField
                Layout.fillWidth: true
                placeholderText: '{"app": "Safari"}'
                font.family: "Menlo"
                font.pixelSize: 13
                text: "{}"
            }

            // --- Имя команды (R8) ---
            Label {
                text: qsTr("Имя команды")
                font.bold: true
                color: Material.foreground
            }

            TextField {
                id: nameField
                Layout.fillWidth: true
                placeholderText: qsTr("Например: Открыть Safari")
                onTextChanged: root.refreshOverwriteHint()
            }

            // --- R6: предупреждение про двуручный жест ---
            Rectangle {
                Layout.fillWidth: true
                visible: root.shouldWarnTwoHands()
                color: "#3a2a14"
                border.color: Material.color(Material.Orange)
                border.width: 1
                radius: 8
                Layout.preferredHeight: warnText.implicitHeight + 16

                Label {
                    id: warnText
                    anchors.fill: parent
                    anchors.margins: 8
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                    color: Material.color(Material.Orange)
                    text: qsTr("⚠ Это «опасное» действие. Рекомендуется привязать его к двуручному жесту, чтобы избежать случайного срабатывания (правило R6).")
                }
            }

            // --- R1: подсказка о перезаписи ---
            Rectangle {
                Layout.fillWidth: true
                visible: root.hasOverwriteCandidate
                color: "#1a2a3a"
                border.color: Material.accent
                border.width: 1
                radius: 8
                Layout.preferredHeight: overwriteText.implicitHeight + 16

                Label {
                    id: overwriteText
                    anchors.fill: parent
                    anchors.margins: 8
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                    color: Material.accent
                    text: qsTr("На этом жесте уже есть команда «") + root.overwriteCandidate
                          + qsTr("». При сохранении она будет заменена (правило R1).")
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Button {
                    text: qsTr("Сохранить привязку")
                    highlighted: true
                    Layout.fillWidth: true
                    enabled: gesturePick.currentIndex >= 0
                             && actionPick.currentIndex >= 0
                             && nameField.text.trim().length > 0
                    onClicked: {
                        root.lastError = ""
                        root.lastInfo = ""
                        var g = root.currentGesture()
                        if (!g) {
                            root.lastError = qsTr("Выберите жест")
                            return
                        }
                        var spec = root.buildActionSpec()
                        if (!spec)
                            return

                        var nameErr = appController.validateCommandName(nameField.text.trim())
                        if (nameErr.length > 0
                                && (!root.hasOverwriteCandidate
                                    || nameField.text.trim() !== root.overwriteCandidate)) {
                            root.lastError = nameErr
                            return
                        }
                        var specErr = appController.validateActionSpec(JSON.stringify(spec))
                        if (specErr.length > 0) {
                            root.lastError = specErr
                            return
                        }

                        var err = appController.saveBinding(
                                    g.label,
                                    nameField.text.trim(),
                                    JSON.stringify(spec))
                        if (err.length > 0) {
                            root.lastError = err
                            return
                        }
                        root.lastInfo = qsTr("Сохранено: «") + g.label
                                + qsTr("»  →  ") + nameField.text.trim()
                        root.refreshDbGestures()
                    }
                }
            }

            Label {
                Layout.fillWidth: true
                visible: root.lastError.length > 0
                wrapMode: Text.WordWrap
                font.pixelSize: 13
                color: Material.color(Material.Red)
                text: root.lastError
            }

            Label {
                Layout.fillWidth: true
                visible: root.lastInfo.length > 0
                wrapMode: Text.WordWrap
                font.pixelSize: 13
                color: Material.color(Material.Green)
                text: root.lastInfo
            }

            Item { Layout.fillHeight: true; Layout.preferredHeight: 12 }
        }
    }
}
