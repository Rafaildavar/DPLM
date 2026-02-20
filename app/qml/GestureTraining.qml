import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Интерфейс обучения жестам - современный дизайн
 * Gesture training interface - modern design
 * 
 * Позволяет пользователю записывать семплы жестов и обучать модель
 * Allows user to record gesture samples and train model
 */
Item {
    id: root
    
    // Состояния обучения / Training states
    property bool isRecording: false
    property int samplesRecorded: 0
    property int targetSamples: 20
    property string currentGesture: ""
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 16
        
        // Заголовок / Header
        Text {
            text: qsTr("Обучение новому жесту")
            font.pixelSize: 20
            font.bold: true
            color: Material.foreground
        }
        
        // Превью камеры с улучшенным дизайном / Enhanced camera preview
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 280
            radius: 20
            
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#1E1E1E" }
                GradientStop { position: 1.0; color: "#2a2a3e" }
            }
            
            border.color: isRecording ? Material.accent : Material.color(Material.Grey, Material.Shade700)
            border.width: isRecording ? 3 : 2
            
            // Анимация пульсации при записи / Pulsing animation when recording
            SequentialAnimation on scale {
                running: isRecording
                loops: Animation.Infinite
                NumberAnimation { to: 1.02; duration: 1000; easing.type: Easing.InOutQuad }
                NumberAnimation { to: 1.0; duration: 1000; easing.type: Easing.InOutQuad }
            }
            
            ColumnLayout {
                anchors.centerIn: parent
                spacing: 16
                
                Rectangle {
                    Layout.preferredWidth: 80
                    Layout.preferredHeight: 80
                    radius: 40
                    color: isRecording ? "#40" + Material.accent.toString().substring(1) : "#20FFFFFF"
                    
                    Text {
                        anchors.centerIn: parent
                        text: "📷"
                        font.pixelSize: 48
                    }
                    
                    SequentialAnimation on rotation {
                        running: isRecording
                        loops: Animation.Infinite
                        NumberAnimation { to: 360; duration: 2000 }
                    }
                }
                
                Text {
                    text: isRecording ? qsTr("Запись...") : qsTr("Камера готова")
                    font.pixelSize: 18
                    font.bold: true
                    color: isRecording ? Material.accent : Material.foreground
                    horizontalAlignment: Text.AlignHCenter
                    Layout.alignment: Qt.AlignHCenter
                }
                
                Text {
                    text: qsTr("(Здесь будет превью с ландмарками)")
                    font.pixelSize: 13
                    color: Material.color(Material.Grey, Material.Shade400)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.alignment: Qt.AlignHCenter
                }
            }
        }
        
        // Прогресс записи с градиентом / Recording progress with gradient
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 120
            radius: 16
            color: "#2a2a3e"
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12
                
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    
                    Text {
                        text: qsTr("Прогресс:")
                        font.pixelSize: 15
                        color: Material.foreground
                    }
                    
                    Text {
                        text: samplesRecorded + " / " + targetSamples
                        font.pixelSize: 16
                        font.bold: true
                        color: Material.accent
                        Layout.fillWidth: true
                    }
                    
                    Rectangle {
                        Layout.preferredWidth: 60
                        Layout.preferredHeight: 30
                        radius: 15
                        color: Material.accent
                        
                        Text {
                            anchors.centerIn: parent
                            text: Math.round((samplesRecorded / targetSamples) * 100) + "%"
                            font.pixelSize: 13
                            font.bold: true
                            color: "white"
                        }
                    }
                }
                
                // Прогресс-бар с градиентом / Progress bar with gradient
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 8
                    radius: 4
                    color: "#1a1a2e"
                    
                    Rectangle {
                        width: parent.width * (samplesRecorded / targetSamples)
                        height: parent.height
                        radius: parent.radius
                        
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: "#4CAF50" }
                            GradientStop { position: 1.0; color: "#388E3C" }
                        }
                        
                        Behavior on width {
                            NumberAnimation { duration: 300; easing.type: Easing.OutCubic }
                        }
                    }
                }
                
                Text {
                    text: samplesRecorded >= targetSamples ? 
                          qsTr("✓ Достаточно семплов! Нажмите 'Обучить'") :
                          qsTr("Выполните жест ещё ") + (targetSamples - samplesRecorded) + qsTr(" раз")
                    font.pixelSize: 13
                    color: samplesRecorded >= targetSamples ? "#4CAF50" : Material.color(Material.Grey, Material.Shade400)
                    Layout.fillWidth: true
                }
            }
        }
        
        // Настройки записи / Recording settings
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 180
            radius: 16
            color: "#2a2a3e"
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12
                
                Label {
                    text: qsTr("Название жеста:")
                    font.pixelSize: 14
                    font.bold: true
                    color: Material.foreground
                }
                
                TextField {
                    id: gestureNameField
                    Layout.fillWidth: true
                    placeholderText: qsTr("Например: swipe_right, thumbs_up")
                    font.pixelSize: 14
                    text: currentGesture
                    enabled: !isRecording
                    Material.accent: Material.accent
                }
                
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    
                    Label {
                        text: qsTr("Целевое количество:")
                        font.pixelSize: 14
                        color: Material.foreground
                    }
                    
                    SpinBox {
                        id: targetSamplesSpinBox
                        from: 10
                        to: 50
                        value: targetSamples
                        stepSize: 5
                        enabled: !isRecording
                        Material.accent: Material.accent
                        
                        onValueChanged: {
                            targetSamples = value
                        }
                    }
                }
                
                CheckBox {
                    id: twoHandsCheckbox
                    text: qsTr("Использовать две руки")
                    font.pixelSize: 14
                    enabled: !isRecording
                    Material.accent: Material.accent
                }
            }
        }
        
        // Кнопки управления с градиентами / Control buttons with gradients
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            
            // Кнопка записи / Record button
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 60
                radius: 16
                
                gradient: Gradient {
                    GradientStop { 
                        position: 0.0
                        color: isRecording ? "#f44336" : Material.accent
                    }
                    GradientStop { 
                        position: 1.0
                        color: isRecording ? "#d32f2f" : Material.primary
                    }
                }
                
                Text {
                    anchors.centerIn: parent
                    text: isRecording ? qsTr("Остановить запись") : qsTr("Начать запись")
                    font.pixelSize: 15
                    font.bold: true
                    color: "white"
                }
                
                MouseArea {
                    anchors.fill: parent
                    enabled: gestureNameField.text.length > 0
                    onClicked: {
                        if (isRecording) {
                            isRecording = false
                            recordingSimulator.stop()
                            console.log("[QML] Запись остановлена")
                        } else {
                            isRecording = true
                            currentGesture = gestureNameField.text
                            samplesRecorded = 0
                            console.log("[QML] Начата запись жеста:", currentGesture)
                            appController.startGestureTraining(currentGesture)
                            recordingSimulator.start()
                        }
                    }
                    
                    states: State {
                        name: "pressed"
                        when: parent.pressed
                        PropertyChanges {
                            target: parent.parent
                            scale: 0.95
                        }
                    }
                    
                    transitions: Transition {
                        NumberAnimation { properties: "scale"; duration: 100 }
                    }
                }
            }
            
            // Кнопка обучения / Train button
            Rectangle {
                Layout.preferredWidth: 120
                Layout.preferredHeight: 60
                radius: 16
                
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#4CAF50" }
                    GradientStop { position: 1.0; color: "#388E3C" }
                }
                
                opacity: (samplesRecorded >= targetSamples && !isRecording) ? 1.0 : 0.5
                
                Text {
                    anchors.centerIn: parent
                    text: qsTr("Обучить")
                    font.pixelSize: 15
                    font.bold: true
                    color: "white"
                }
                
                MouseArea {
                    anchors.fill: parent
                    enabled: samplesRecorded >= targetSamples && !isRecording
                    onClicked: {
                        console.log("[QML] Начато обучение модели для жеста:", currentGesture)
                        trainingProgress.visible = true
                        trainingSimulator.start()
                    }
                    
                    states: State {
                        name: "pressed"
                        when: parent.pressed
                        PropertyChanges {
                            target: parent.parent
                            scale: 0.95
                        }
                    }
                    
                    transitions: Transition {
                        NumberAnimation { properties: "scale"; duration: 100 }
                    }
                }
            }
            
            // Кнопка сброса / Reset button
            Rectangle {
                Layout.preferredWidth: 100
                Layout.preferredHeight: 60
                radius: 16
                color: "#757575"
                
                opacity: isRecording ? 0.5 : 1.0
                
                Text {
                    anchors.centerIn: parent
                    text: qsTr("Сброс")
                    font.pixelSize: 15
                    font.bold: true
                    color: "white"
                }
                
                MouseArea {
                    anchors.fill: parent
                    enabled: !isRecording
                    onClicked: {
                        samplesRecorded = 0
                        gestureNameField.text = ""
                        currentGesture = ""
                        console.log("[QML] Сброс обучения")
                    }
                    
                    states: State {
                        name: "pressed"
                        when: parent.pressed
                        PropertyChanges {
                            target: parent.parent
                            scale: 0.95
                        }
                    }
                    
                    transitions: Transition {
                        NumberAnimation { properties: "scale"; duration: 100 }
                    }
                }
            }
        }
        
        // Индикатор обучения модели / Training indicator
        Rectangle {
            id: trainingProgress
            Layout.fillWidth: true
            Layout.preferredHeight: 70
            radius: 16
            visible: false
            
            gradient: Gradient {
                GradientStop { position: 0.0; color: Material.accent }
                GradientStop { position: 1.0; color: Material.primary }
            }
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 16
                
                BusyIndicator {
                    Layout.preferredWidth: 40
                    Layout.preferredHeight: 40
                    running: trainingProgress.visible
                    Material.accent: "white"
                }
                
                Text {
                    text: qsTr("Обучение модели... Пожалуйста, подождите")
                    font.pixelSize: 15
                    font.bold: true
                    color: "white"
                }
            }
        }
    }
    
    // Таймер для симуляции записи семплов / Timer to simulate sample recording
    Timer {
        id: recordingSimulator
        interval: 1500
        repeat: true
        running: false
        
        onTriggered: {
            if (isRecording && samplesRecorded < targetSamples) {
                samplesRecorded++
                console.log("[QML] Записан семпл", samplesRecorded)
            }
            
            if (samplesRecorded >= targetSamples) {
                stop()
            }
        }
    }
    
    // Таймер для симуляции обучения модели / Timer to simulate model training
    Timer {
        id: trainingSimulator
        interval: 3000
        repeat: false
        running: false
        
        onTriggered: {
            trainingProgress.visible = false
            successDialog.open()
        }
    }
    
    // Диалог успешного обучения / Success dialog
    Rectangle {
        id: successDialog
        anchors.fill: parent
        color: "#80000000"
        visible: false
        opacity: 0
        
        function open() {
            visible = true
            openAnimation.start()
        }
        
        function close() {
            closeAnimation.start()
        }
        
        SequentialAnimation {
            id: openAnimation
            NumberAnimation {
                target: successDialog
                property: "opacity"
                to: 1.0
                duration: 300
            }
        }
        
        SequentialAnimation {
            id: closeAnimation
            NumberAnimation {
                target: successDialog
                property: "opacity"
                to: 0
                duration: 300
            }
            PropertyAction {
                target: successDialog
                property: "visible"
                value: false
            }
        }
        
        MouseArea {
            anchors.fill: parent
            onClicked: successDialog.close()
        }
        
        Rectangle {
            anchors.centerIn: parent
            width: 400
            height: 250
            radius: 20
            color: "#2a2a3e"
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 24
                spacing: 16
                
                Rectangle {
                    Layout.preferredWidth: 60
                    Layout.preferredHeight: 60
                    Layout.alignment: Qt.AlignHCenter
                    radius: 30
                    color: "#4CAF50"
                    
                    Text {
                        anchors.centerIn: parent
                        text: "✓"
                        font.pixelSize: 36
                        color: "white"
                        font.bold: true
                    }
                }
                
                Text {
                    text: qsTr("Обучение завершено!")
                    font.pixelSize: 20
                    font.bold: true
                    color: Material.foreground
                    Layout.alignment: Qt.AlignHCenter
                }
                
                Text {
                    text: qsTr("Модель для жеста '") + currentGesture + qsTr("' успешно обучена!\n\nТеперь вы можете назначить этот жест команде.")
                    font.pixelSize: 14
                    color: Material.color(Material.Grey, Material.Shade400)
                    Layout.alignment: Qt.AlignHCenter
                    horizontalAlignment: Text.AlignHCenter
                    wrapMode: Text.WordWrap
                }
                
                Item {
                    Layout.fillHeight: true
                }
                
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 45
                    radius: 22
                    color: Material.accent
                    
                    Text {
                        anchors.centerIn: parent
                        text: qsTr("OK")
                        font.pixelSize: 15
                        font.bold: true
                        color: "white"
                    }
                    
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            samplesRecorded = 0
                            gestureNameField.text = ""
                            currentGesture = ""
                            successDialog.close()
                        }
                    }
                }
            }
        }
    }
}
