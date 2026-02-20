import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Панель настроек приложения
 * Application settings panel
 * 
 * Модальное окно с настройками камеры, голоса, аватара, горячих клавиш
 * Modal window with camera, voice, avatar, hotkey settings
 */
Rectangle {
    id: root
    
    color: Material.backgroundColor
    radius: 12
    border.color: Material.accent
    border.width: 2
    
    // Overlay для модальности / Overlay for modality
    Rectangle {
        id: overlay
        anchors.fill: parent.parent
        color: "#80000000"
        visible: root.visible
        
        MouseArea {
            anchors.fill: parent
            onClicked: {
                root.visible = false
            }
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16
        
        // Заголовок / Header
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            
            Text {
                text: qsTr("⚙ Настройки")
                font.pixelSize: 20
                font.bold: true
                color: Material.foreground
                Layout.fillWidth: true
            }
            
            Button {
                text: "✕"
                font.pixelSize: 18
                flat: true
                Layout.preferredWidth: 40
                Layout.preferredHeight: 40
                
                onClicked: {
                    root.visible = false
                }
            }
        }
        
        // Контент с прокруткой / Scrollable content
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            
            ColumnLayout {
                width: parent.width
                spacing: 20
                
                // Секция: Камера / Section: Camera
                GroupBox {
                    Layout.fillWidth: true
                    title: qsTr("Камера")
                    font.pixelSize: 16
                    
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 12
                        
                        Label {
                            text: qsTr("Выбор камеры:")
                            font.pixelSize: 14
                        }
                        
                        ComboBox {
                            id: cameraCombo
                            Layout.fillWidth: true
                            model: ["Камера по умолчанию", "FaceTime HD Camera", "USB Camera"]
                            font.pixelSize: 14
                        }
                        
                        CheckBox {
                            id: showLandmarksCheckbox
                            text: qsTr("Показывать ландмарки на превью")
                            font.pixelSize: 14
                            checked: true
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            Label {
                                text: qsTr("Качество детекции:")
                                font.pixelSize: 14
                            }
                            
                            Slider {
                                id: detectionQualitySlider
                                Layout.fillWidth: true
                                from: 0
                                to: 2
                                value: 1
                                stepSize: 1
                                snapMode: Slider.SnapAlways
                            }
                            
                            Label {
                                text: ["Низкое", "Среднее", "Высокое"][detectionQualitySlider.value]
                                font.pixelSize: 12
                                color: Material.color(Material.Grey)
                            }
                        }
                    }
                }
                
                // Секция: Голосовой помощник / Section: Voice Assistant
                GroupBox {
                    id: voiceAssistantGroup
                    Layout.fillWidth: true
                    title: qsTr("Голосовой помощник")
                    font.pixelSize: 16
                    
                    ColumnLayout {
                        id: voiceAssistantLayout
                        anchors.fill: parent
                        spacing: 12
                        
                        // Функция обновления списка голосов / Function to update voices list
                        function updateVoicesList() {
                            var langMap = {"Русский": "ru", "English": "en", "Deutsch": "de"}
                            var lang = langMap[assistantLanguageCombo.currentText]
                            
                            var voices = appController.getAvailableVoicesByLanguage(lang)
                            voicesModel.clear()
                            
                            for (var i = 0; i < voices.length; i++) {
                                voicesModel.append({
                                    "id": voices[i].id,
                                    "name": voices[i].name,
                                    "language": voices[i].language
                                })
                            }
                            
                            // Установить текущий голос если помощник активен / Set current voice if assistant active
                            if (appController.isVoiceAssistantActive && voicesModel.count > 0) {
                                var currentVoice = appController.getCurrentVoice()
                                if (currentVoice && currentVoice.id) {
                                    for (var j = 0; j < voicesModel.count; j++) {
                                        if (voicesModel.get(j).id === currentVoice.id) {
                                            voiceComboBox.currentIndex = j
                                            break
                                        }
                                    }
                                } else if (voicesModel.count > 0) {
                                    // Выбрать первый голос по умолчанию / Select first voice by default
                                    voiceComboBox.currentIndex = 0
                                }
                            }
                        }
                        
                        // Статус голосового помощника / Voice assistant status
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 40
                            color: appController.isVoiceAssistantActive ? 
                                   Material.color(Material.Green, Material.Shade700) : 
                                   Material.color(Material.Grey, Material.Shade700)
                            radius: 8
                            
                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 8
                                
                                Text {
                                    text: appController.isVoiceAssistantActive ? 
                                          qsTr("● Активен") : qsTr("○ Неактивен")
                                    font.pixelSize: 14
                                    color: "white"
                                    Layout.fillWidth: true
                                }
                                
                                Text {
                                    text: appController.getVoiceAssistantState()
                                    font.pixelSize: 12
                                    color: "white"
                                    opacity: 0.8
                                }
                            }
                        }
                        
                        CheckBox {
                            id: enableTTSCheckbox
                            text: qsTr("Включить озвучку (TTS)")
                            font.pixelSize: 14
                            checked: true
                        }
                        
                        Label {
                            text: qsTr("Язык помощника:")
                            font.pixelSize: 14
                        }
                        
                        ComboBox {
                            id: assistantLanguageCombo
                            Layout.fillWidth: true
                            model: ["Русский", "English", "Deutsch"]
                            font.pixelSize: 14
                            currentIndex: 0
                            
                            onCurrentIndexChanged: {
                                // Обновить список голосов при смене языка / Update voices on language change
                                if (assistantLanguageCombo.currentIndex >= 0) {
                                    voiceAssistantLayout.updateVoicesList()
                                }
                            }
                        }
                        
                        // Выбор голоса / Voice selection
                        Label {
                            text: qsTr("Голос помощника:")
                            font.pixelSize: 14
                            visible: voiceComboBox.count > 0
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            visible: voiceComboBox.count > 0
                            
                            ComboBox {
                                id: voiceComboBox
                                Layout.fillWidth: true
                                font.pixelSize: 14
                                
                                // Модель будет заполнена через JavaScript / Model will be filled via JavaScript
                                model: ListModel {
                                    id: voicesModel
                                }
                                
                                // Отображение имени голоса / Display voice name
                                textRole: "name"
                                
                                onCurrentIndexChanged: {
                                    if (currentIndex >= 0 && currentIndex < voicesModel.count) {
                                        var voiceId = voicesModel.get(currentIndex).id
                                        if (appController.isVoiceAssistantActive) {
                                            appController.setVoiceAssistantVoice(voiceId)
                                        }
                                    }
                                }
                            }
                            
                            // Кнопка предпросмотра голоса / Voice preview button
                            Button {
                                text: "▶"
                                font.pixelSize: 12
                                Layout.preferredWidth: 36
                                Layout.preferredHeight: 36
                                flat: true
                                Material.accent: Material.Blue
                                
                                onClicked: {
                                    if (voiceComboBox.currentIndex >= 0) {
                                        var voiceId = voicesModel.get(voiceComboBox.currentIndex).id
                                        appController.setVoiceAssistantVoice(voiceId)
                                        appController.previewVoiceAssistantVoice("Привет, это тест голоса")
                                    }
                                }
                            }
                        }
                        
                        // Инициализация при загрузке / Initialize on load
                        Component.onCompleted: {
                            voiceAssistantLayout.updateVoicesList()
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            Label {
                                text: qsTr("Скорость речи:")
                                font.pixelSize: 14
                            }
                            
                            Slider {
                                id: speechRateSlider
                                Layout.fillWidth: true
                                from: 50
                                to: 300
                                value: 150
                                stepSize: 10
                                
                                onValueChanged: {
                                    if (appController.isVoiceAssistantActive) {
                                        appController.setVoiceAssistantSpeechRate(value)
                                    }
                                }
                            }
                            
                            Label {
                                text: Math.round(speechRateSlider.value) + " WPM"
                                font.pixelSize: 12
                                color: Material.color(Material.Grey)
                                Layout.preferredWidth: 60
                            }
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            Label {
                                text: qsTr("Громкость:")
                                font.pixelSize: 14
                            }
                            
                            Slider {
                                id: volumeSlider
                                Layout.fillWidth: true
                                from: 0.0
                                to: 1.0
                                value: 0.9
                                stepSize: 0.1
                                
                                onValueChanged: {
                                    if (appController.isVoiceAssistantActive) {
                                        appController.setVoiceAssistantVolume(value)
                                    }
                                }
                            }
                            
                            Label {
                                text: Math.round(volumeSlider.value * 100) + "%"
                                font.pixelSize: 12
                                color: Material.color(Material.Grey)
                                Layout.preferredWidth: 50
                            }
                        }
                        
                        CheckBox {
                            id: enableWakeWordCheckbox
                            text: qsTr("Активация по ключевому слову")
                            font.pixelSize: 14
                            checked: true
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 62
                            radius: 10
                            color: "#1f3b4d"

                            Text {
                                anchors.fill: parent
                                anchors.margins: 10
                                text: qsTr("Как использовать: нажмите 'Запустить', затем скажите 'ассистент' + команду. Команды: привет, помощь, статус, открыть браузер.")
                                color: "#E1F5FE"
                                wrapMode: Text.WordWrap
                                font.pixelSize: 12
                            }
                        }
                        
                        // Кнопки управления / Control buttons
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            
                            Button {
                                id: startVoiceAssistantBtn
                                text: qsTr("▶ Запустить")
                                Layout.fillWidth: true
                                Material.accent: Material.Green
                                enabled: !appController.isVoiceAssistantActive
                                
                            onClicked: {
                                var langMap = {"Русский": "ru", "English": "en", "Deutsch": "de"}
                                var lang = langMap[assistantLanguageCombo.currentText]
                                var success = appController.startVoiceAssistant(lang, enableWakeWordCheckbox.checked, enableTTSCheckbox.checked)
                                
                                if (success) {
                                    // Обновить список голосов и установить выбранный / Update voices and set selected
                                    voiceAssistantLayout.updateVoicesList()
                                    if (voiceComboBox.count > 0 && voiceComboBox.currentIndex >= 0) {
                                        var voiceId = voicesModel.get(voiceComboBox.currentIndex).id
                                        appController.setVoiceAssistantVoice(voiceId)
                                    } else if (voiceComboBox.count > 0) {
                                        // Выбрать первый голос если ничего не выбрано / Select first voice if none selected
                                        voiceComboBox.currentIndex = 0
                                        var voiceId = voicesModel.get(0).id
                                        appController.setVoiceAssistantVoice(voiceId)
                                    }
                                    // Применить настройки скорости и громкости / Apply rate and volume settings
                                    appController.setVoiceAssistantSpeechRate(speechRateSlider.value)
                                    appController.setVoiceAssistantVolume(volumeSlider.value)
                                }
                            }
                            }
                            
                            Button {
                                id: stopVoiceAssistantBtn
                                text: qsTr("■ Остановить")
                                Layout.fillWidth: true
                                Material.accent: Material.Red
                                enabled: appController.isVoiceAssistantActive
                                
                                onClicked: {
                                    appController.stopVoiceAssistant()
                                }
                            }
                        }
                        
                        // Индикатор прослушивания / Listening indicator
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 4
                            color: Material.color(Material.Grey, Material.Shade800)
                            radius: 2
                            visible: appController.isVoiceAssistantActive
                            
                            Rectangle {
                                anchors.fill: parent
                                color: Material.accent
                                radius: 2
                                
                                SequentialAnimation on opacity {
                                    running: appController.isVoiceAssistantActive
                                    loops: Animation.Infinite
                                    NumberAnimation { to: 0.3; duration: 1000 }
                                    NumberAnimation { to: 1.0; duration: 1000 }
                                }
                            }
                        }
                    }
                }
                
                // Секция: Аватар / Section: Avatar
                GroupBox {
                    Layout.fillWidth: true
                    title: qsTr("Аватар-ассистент")
                    font.pixelSize: 16
                    
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 12
                        
                        CheckBox {
                            id: enableAvatarCheckbox
                            text: qsTr("Показывать аватара")
                            font.pixelSize: 14
                            checked: false
                        }
                        
                        Label {
                            text: qsTr("Стиль аватара:")
                            font.pixelSize: 14
                            enabled: enableAvatarCheckbox.checked
                        }
                        
                        ComboBox {
                            id: avatarStyleCombo
                            Layout.fillWidth: true
                            model: ["Минималистичный", "Анимированный", "3D персонаж"]
                            font.pixelSize: 14
                            enabled: enableAvatarCheckbox.checked
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            enabled: enableAvatarCheckbox.checked
                            
                            Label {
                                text: qsTr("Прозрачность:")
                                font.pixelSize: 14
                            }
                            
                            Slider {
                                id: avatarOpacitySlider
                                Layout.fillWidth: true
                                from: 0.3
                                to: 1.0
                                value: 0.9
                                stepSize: 0.1
                            }
                            
                            Label {
                                text: Math.round(avatarOpacitySlider.value * 100) + "%"
                                font.pixelSize: 12
                                color: Material.color(Material.Grey)
                            }
                        }
                    }
                }
                
                // Секция: Горячие клавиши / Section: Hotkeys
                GroupBox {
                    Layout.fillWidth: true
                    title: qsTr("Горячие клавиши")
                    font.pixelSize: 16
                    
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 12
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            Label {
                                text: qsTr("Вкл/Выкл панели:")
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }
                            
                            TextField {
                                id: togglePanelHotkey
                                Layout.fillWidth: true
                                text: "Cmd+Shift+G"
                                font.pixelSize: 14
                                readOnly: true
                            }
                            
                            Button {
                                text: qsTr("Изменить")
                                font.pixelSize: 12
                                flat: true
                            }
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            Label {
                                text: qsTr("Вызов аватара:")
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }
                            
                            TextField {
                                id: avatarHotkey
                                Layout.fillWidth: true
                                text: "Cmd+Shift+A"
                                font.pixelSize: 14
                                readOnly: true
                            }
                            
                            Button {
                                text: qsTr("Изменить")
                                font.pixelSize: 12
                                flat: true
                            }
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            Label {
                                text: qsTr("Голосовая команда:")
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }
                            
                            TextField {
                                id: voiceHotkey
                                Layout.fillWidth: true
                                text: "Cmd+Shift+V"
                                font.pixelSize: 14
                                readOnly: true
                            }
                            
                            Button {
                                text: qsTr("Изменить")
                                font.pixelSize: 12
                                flat: true
                            }
                        }
                    }
                }
                
                // Секция: Дополнительно / Section: Advanced
                GroupBox {
                    Layout.fillWidth: true
                    title: qsTr("Дополнительно")
                    font.pixelSize: 16
                    
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 12
                        
                        CheckBox {
                            id: autoStartCheckbox
                            text: qsTr("Запускать при входе в систему")
                            font.pixelSize: 14
                            checked: false
                        }
                        
                        CheckBox {
                            id: minimizeToTrayCheckbox
                            text: qsTr("Сворачивать в трей")
                            font.pixelSize: 14
                            checked: true
                        }
                        
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            
                            Label {
                                text: qsTr("Окно сглаживания (кадры):")
                                font.pixelSize: 14
                            }
                            
                            SpinBox {
                                id: smoothingWindowSpinBox
                                from: 5
                                to: 60
                                value: 30
                                stepSize: 5
                            }
                        }
                    }
                }
            }
        }
        
        // Кнопки действий / Action buttons
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            
            Button {
                text: qsTr("Сбросить по умолчанию")
                font.pixelSize: 14
                Layout.preferredWidth: 180
                Material.background: Material.color(Material.Grey)
                Material.foreground: "white"
                
                onClicked: {
                    // Сброс всех настроек
                    // Reset all settings
                    console.log("[QML] Сброс настроек по умолчанию")
                    cameraCombo.currentIndex = 0
                    ttsLanguageCombo.currentIndex = 0
                    speechRateSlider.value = 1.0
                    detectionQualitySlider.value = 1
                    enableTTSCheckbox.checked = true
                    enableSTTCheckbox.checked = false
                    enableAvatarCheckbox.checked = false
                    smoothingWindowSpinBox.value = 30
                }
            }
            
            Item {
                Layout.fillWidth: true
            }
            
            Button {
                text: qsTr("Отмена")
                font.pixelSize: 14
                Layout.preferredWidth: 100
                Material.background: Material.color(Material.Grey)
                Material.foreground: "white"
                
                onClicked: {
                    root.visible = false
                }
            }
            
            Button {
                text: qsTr("Сохранить")
                font.pixelSize: 14
                font.bold: true
                Layout.preferredWidth: 120
                Material.background: Material.accent
                Material.foreground: "white"
                
                onClicked: {
                    // Сохранение настроек в БД
                    // Save settings to DB
                    console.log("[QML] Сохранение настроек")
                    console.log("  - Камера:", cameraCombo.currentText)
                    console.log("  - TTS:", enableTTSCheckbox.checked)
                    console.log("  - Язык:", ttsLanguageCombo.currentText)
                    console.log("  - Скорость речи:", speechRateSlider.value)
                    console.log("  - STT:", enableSTTCheckbox.checked)
                    console.log("  - Аватар:", enableAvatarCheckbox.checked)
                    console.log("  - Окно сглаживания:", smoothingWindowSpinBox.value)
                    
                    // TODO: сохранение в app/models/database.py (таблица settings)
                    // TODO: save to app/models/database.py (settings table)
                    
                    root.visible = false
                }
            }
        }
    }
}

