import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window
import QtGraphicalEffects

/**
 * Главное окно приложения DPLM - современный интерфейс
 * Main DPLM application window - modern interface
 * 
 * Material Design 3 с улучшенной визуализацией
 * Material Design 3 with enhanced visualization
 */
Window {
    id: mainWindow
    
    // Настройки окна / Window settings
    width: 450
    height: 700
    minimumWidth: 400
    minimumHeight: 600
    visible: true
    title: "DPLM - Gesture & Voice Assistant"
    
    // Material Design 3 тема / Material Design 3 theme
    Material.theme: Material.Dark
    Material.accent: Material.Cyan
    Material.primary: Material.Cyan
    
    // Градиентный фон / Gradient background
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#1a1a2e" }
            GradientStop { position: 1.0; color: "#16213e" }
        }
    }
    
    // Основной контейнер / Main container
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20
        
        // Заголовок с градиентом / Header with gradient
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 100
            radius: 20
            
            gradient: Gradient {
                GradientStop { position: 0.0; color: Material.accent }
                GradientStop { position: 1.0; color: Material.primary }
            }
            
            // Тень / Shadow
            layer.enabled: true
            layer.effect: DropShadow {
                transparentBorder: true
                horizontalOffset: 0
                verticalOffset: 4
                radius: 12
                samples: 17
                color: "#40000000"
            }
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 16
                
                // Иконка статуса с анимацией / Animated status icon
                Rectangle {
                    Layout.preferredWidth: 60
                    Layout.preferredHeight: 60
                    radius: 30
                    color: appController.isRecognizing ? "#4CAF50" : "#757575"
                    
                    // Пульсирующая анимация / Pulsing animation
                    SequentialAnimation on scale {
                        running: appController.isRecognizing
                        loops: Animation.Infinite
                        NumberAnimation { to: 1.1; duration: 800; easing.type: Easing.InOutQuad }
                        NumberAnimation { to: 1.0; duration: 800; easing.type: Easing.InOutQuad }
                    }
                    
                    layer.enabled: true
                    layer.effect: DropShadow {
                        transparentBorder: true
                        horizontalOffset: 0
                        verticalOffset: 2
                        radius: 8
                        samples: 17
                        color: appController.isRecognizing ? "#804CAF50" : "#40000000"
                    }
                    
                    Text {
                        anchors.centerIn: parent
                        text: appController.isRecognizing ? "●" : "○"
                        font.pixelSize: 28
                        color: "white"
                        font.bold: true
                    }
                }
                
                // Текст статуса / Status text
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    
                    Text {
                        text: "DPLM Assistant"
                        font.pixelSize: 22
                        font.bold: true
                        color: "white"
                        font.family: "SF Pro Display, Segoe UI, Arial"
                    }
                    
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        
                        Text {
                            text: appController.status
                            font.pixelSize: 14
                            color: "#E0E0E0"
                            Layout.fillWidth: true
                        }
                        
                        // Индикатор голосового помощника с анимацией / Animated voice assistant indicator
                        Rectangle {
                            Layout.preferredWidth: 10
                            Layout.preferredHeight: 10
                            radius: 5
                            color: appController.isVoiceAssistantActive ? "#4CAF50" : "#757575"
                            
                            SequentialAnimation on opacity {
                                running: appController.isVoiceAssistantActive
                                loops: Animation.Infinite
                                NumberAnimation { to: 0.3; duration: 1000; easing.type: Easing.InOutQuad }
                                NumberAnimation { to: 1.0; duration: 1000; easing.type: Easing.InOutQuad }
                            }
                            
                            SequentialAnimation on scale {
                                running: appController.isVoiceAssistantActive
                                loops: Animation.Infinite
                                NumberAnimation { to: 1.2; duration: 1000; easing.type: Easing.InOutQuad }
                                NumberAnimation { to: 1.0; duration: 1000; easing.type: Easing.InOutQuad }
                            }
                        }
                    }
                }
                
                // Кнопка настроек с эффектом / Settings button with effect
                Button {
                    Layout.preferredWidth: 50
                    Layout.preferredHeight: 50
                    text: "⚙"
                    font.pixelSize: 22
                    flat: true
                    Material.foreground: "white"
                    
                    background: Rectangle {
                        color: parent.hovered ? "#40FFFFFF" : "transparent"
                        radius: 25
                        Behavior on color {
                            ColorAnimation { duration: 200 }
                        }
                    }
                    
                    onClicked: {
                        settingsPanel.visible = true
                    }
                }
            }
        }
        
        // Вкладки с улучшенным дизайном / Enhanced tabs design
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 50
            color: "#2a2a3e"
            radius: 12
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 4
                spacing: 4
                
                // Кнопка "Команды" / "Commands" button
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 10
                    color: tabBar.currentIndex === 0 ? Material.accent : "transparent"
                    
                    Behavior on color {
                        ColorAnimation { duration: 300 }
                    }
                    
                    Text {
                        anchors.centerIn: parent
                        text: qsTr("Команды")
                        font.pixelSize: 15
                        font.bold: tabBar.currentIndex === 0
                        color: tabBar.currentIndex === 0 ? "white" : Material.foreground
                    }
                    
                    MouseArea {
                        anchors.fill: parent
                        onClicked: tabBar.currentIndex = 0
                    }
                }
                
                // Кнопка "Обучение" / "Training" button
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 10
                    color: tabBar.currentIndex === 1 ? Material.accent : "transparent"
                    
                    Behavior on color {
                        ColorAnimation { duration: 300 }
                    }
                    
                    Text {
                        anchors.centerIn: parent
                        text: qsTr("Обучение")
                        font.pixelSize: 15
                        font.bold: tabBar.currentIndex === 1
                        color: tabBar.currentIndex === 1 ? "white" : Material.foreground
                    }
                    
                    MouseArea {
                        anchors.fill: parent
                        onClicked: tabBar.currentIndex = 1
                    }
                }
            }
        }
        
        // Контент вкладок с анимацией / Tab content with animation
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            StackLayout {
                id: contentStack
                anchors.fill: parent
                currentIndex: tabBar.currentIndex
                
                // Вкладка "Команды" / "Commands" tab
                Item {
                    CommandsPanel {
                        id: commandsPanel
                        anchors.fill: parent
                    }
                }
                
                // Вкладка "Обучение" / "Training" tab
                Item {
                    GestureTraining {
                        id: gestureTraining
                        anchors.fill: parent
                    }
                }
            }
            
            // Анимация перехода между вкладками / Tab transition animation
            Behavior on opacity {
                NumberAnimation { duration: 300; easing.type: Easing.InOutQuad }
            }
        }
        
        // Панель управления распознаванием с градиентом / Recognition control panel with gradient
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 80
            radius: 16
            color: "#2a2a3e"
            
            layer.enabled: true
            layer.effect: DropShadow {
                transparentBorder: true
                horizontalOffset: 0
                verticalOffset: 4
                radius: 12
                samples: 17
                color: "#30000000"
            }
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12
                
                // Кнопка старт/стоп с градиентом / Start/stop button with gradient
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 12
                    
                    gradient: Gradient {
                        GradientStop { 
                            position: 0.0
                            color: appController.isRecognizing ? "#f44336" : "#4CAF50"
                        }
                        GradientStop { 
                            position: 1.0
                            color: appController.isRecognizing ? "#d32f2f" : "#388E3C"
                        }
                    }
                    
                    layer.enabled: true
                    layer.effect: DropShadow {
                        transparentBorder: true
                        horizontalOffset: 0
                        verticalOffset: 2
                        radius: 8
                        samples: 17
                        color: appController.isRecognizing ? "#80f44336" : "#804CAF50"
                    }
                    
                    Text {
                        anchors.centerIn: parent
                        text: appController.isRecognizing ? qsTr("Остановить") : qsTr("Начать распознавание")
                        font.pixelSize: 16
                        font.bold: true
                        color: "white"
                    }
                    
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            if (appController.isRecognizing) {
                                appController.stopRecognition()
                            } else {
                                appController.startRecognition()
                            }
                        }
                    }
                    
                    // Эффект нажатия / Press effect
                    states: State {
                        name: "pressed"
                        when: mouseArea.pressed
                        PropertyChanges {
                            target: parent
                            scale: 0.95
                        }
                    }
                    
                    transitions: Transition {
                        NumberAnimation { properties: "scale"; duration: 100 }
                    }
                    
                    MouseArea {
                        id: mouseArea
                        anchors.fill: parent
                        onClicked: {
                            if (appController.isRecognizing) {
                                appController.stopRecognition()
                            } else {
                                appController.startRecognition()
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Панель настроек (модальное окно) / Settings panel (modal)
    SettingsPanel {
        id: settingsPanel
        visible: false
        width: Math.min(parent.width * 0.95, 600)
        height: Math.min(parent.height * 0.95, 800)
        x: (parent.width - width) / 2
        y: (parent.height - height) / 2
        
        Behavior on opacity {
            NumberAnimation { duration: 300 }
        }
        
        Behavior on scale {
            NumberAnimation { duration: 300; easing.type: Easing.OutBack }
        }
        
        states: State {
            name: "visible"
            when: settingsPanel.visible
            PropertyChanges {
                target: settingsPanel
                opacity: 1
                scale: 1
            }
        }
        
        PropertyChanges {
            target: settingsPanel
            opacity: 0
            scale: 0.9
        }
    }
    
    // Обработчики событий от контроллера / Controller event handlers
    Connections {
        target: appController
        
        function onGestureDetected(gesture) {
            console.log("[QML] Обнаружен жест:", gesture)
            statusNotification.show("Жест: " + gesture)
        }
        
        function onCommandExecuted(command) {
            console.log("[QML] Выполнена команда:", command)
            statusNotification.show("Команда выполнена: " + command)
        }
    }
    
    // Компонент уведомлений с улучшенным дизайном / Enhanced notification component
    Rectangle {
        id: statusNotification
        width: parent.width * 0.85
        height: 70
        x: (parent.width - width) / 2
        y: parent.height - height - 30
        radius: 16
        opacity: 0
        scale: 0.8
        
        gradient: Gradient {
            GradientStop { position: 0.0; color: Material.accent }
            GradientStop { position: 1.0; color: Material.primary }
        }
        
        layer.enabled: true
        layer.effect: DropShadow {
            transparentBorder: true
            horizontalOffset: 0
            verticalOffset: 4
            radius: 16
            samples: 17
            color: "#60000000"
        }
        
        property string message: ""
        
        RowLayout {
            anchors.fill: parent
            anchors.margins: 16
            spacing: 12
            
            Rectangle {
                Layout.preferredWidth: 40
                Layout.preferredHeight: 40
                radius: 20
                color: "#40FFFFFF"
                
                Text {
                    anchors.centerIn: parent
                    text: "✓"
                    font.pixelSize: 20
                    color: "white"
                    font.bold: true
                }
            }
            
            Text {
                Layout.fillWidth: true
                text: parent.parent.message
                font.pixelSize: 15
                font.bold: true
                color: "white"
                wrapMode: Text.WordWrap
            }
        }
        
        function show(msg) {
            message = msg
            showAnimation.start()
        }
        
        SequentialAnimation {
            id: showAnimation
            ParallelAnimation {
                NumberAnimation {
                    target: statusNotification
                    property: "opacity"
                    to: 1.0
                    duration: 300
                    easing.type: Easing.OutCubic
                }
                NumberAnimation {
                    target: statusNotification
                    property: "scale"
                    to: 1.0
                    duration: 300
                    easing.type: Easing.OutBack
                }
            }
            PauseAnimation { duration: 2500 }
            ParallelAnimation {
                NumberAnimation {
                    target: statusNotification
                    property: "opacity"
                    to: 0
                    duration: 300
                    easing.type: Easing.InCubic
                }
                NumberAnimation {
                    target: statusNotification
                    property: "scale"
                    to: 0.8
                    duration: 300
                    easing.type: Easing.InBack
                }
            }
        }
    }
}
