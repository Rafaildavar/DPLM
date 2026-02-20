import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts

/**
 * Панель управления командами - современный дизайн
 * Commands management panel - modern design
 * 
 * Показывает список доступных команд и позволяет добавлять новые
 * Shows list of available commands and allows adding new ones
 */
Item {
    id: root
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 16
        
        // Заголовок и кнопка добавления / Header and add button
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            
            Text {
                text: qsTr("Доступные команды")
                font.pixelSize: 18
                font.bold: true
                color: Material.foreground
                Layout.fillWidth: true
            }
            
            // Кнопка добавления с градиентом / Add button with gradient
            Rectangle {
                Layout.preferredWidth: 120
                Layout.preferredHeight: 40
                radius: 20
                
                gradient: Gradient {
                    GradientStop { position: 0.0; color: Material.accent }
                    GradientStop { position: 1.0; color: Material.primary }
                }
                
                Text {
                    anchors.centerIn: parent
                    text: "+ Добавить"
                    font.pixelSize: 14
                    font.bold: true
                    color: "white"
                }
                
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        addCommandDialog.open()
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
        
        // Список команд / Commands list
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            
            ScrollBar.vertical.policy: ScrollBar.AlwaysOn
            ScrollBar.vertical.interactive: true
            
            ListView {
                id: commandsList
                spacing: 12
                clip: true
                
                // Модель команд / Commands model
                model: ListModel {
                    id: commandsModel
                    
                    Component.onCompleted: {
                        var commands = appController.getAvailableCommands()
                        for (var i = 0; i < commands.length; i++) {
                            commandsModel.append(commands[i])
                        }
                    }
                }
                
                // Делегат команды с улучшенным дизайном / Enhanced command delegate
                delegate: Rectangle {
                    id: commandCard
                    width: commandsList.width
                    height: 100
                    radius: 16
                    color: "#2a2a3e"
                    
                    // Градиент при наведении / Gradient on hover
                    Rectangle {
                        anchors.fill: parent
                        radius: parent.radius
                        gradient: Gradient {
                            GradientStop { 
                                position: 0.0
                                color: mouseArea.containsMouse ? "#3a3a4e" : "transparent"
                            }
                            GradientStop { 
                                position: 1.0
                                color: mouseArea.containsMouse ? "#2a2a3e" : "transparent"
                            }
                        }
                        opacity: mouseArea.containsMouse ? 1 : 0
                        
                        Behavior on opacity {
                            NumberAnimation { duration: 200 }
                        }
                    }
                    
                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 16
                        
                        // Иконка команды с градиентом / Command icon with gradient
                        Rectangle {
                            Layout.preferredWidth: 60
                            Layout.preferredHeight: 60
                            radius: 30
                            
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: Material.accent }
                                GradientStop { position: 1.0; color: Material.primary }
                            }
                            
                            Text {
                                anchors.centerIn: parent
                                text: "⚡"
                                font.pixelSize: 28
                            }
                        }
                        
                        // Информация о команде / Command info
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            
                            Text {
                                text: model.name
                                font.pixelSize: 16
                                font.bold: true
                                color: Material.foreground
                            }
                            
                            Text {
                                text: qsTr("Жест: ") + (model.gesture || "не назначен")
                                font.pixelSize: 13
                                color: Material.color(Material.Grey, Material.Shade400)
                            }
                        }
                        
                        // Кнопки действий / Action buttons
                        RowLayout {
                            spacing: 8
                            
                            // Кнопка выполнить / Execute button
                            Rectangle {
                                Layout.preferredWidth: 44
                                Layout.preferredHeight: 44
                                radius: 22
                                color: "#4CAF50"
                                
                                Text {
                                    anchors.centerIn: parent
                                    text: "▶"
                                    font.pixelSize: 16
                                    color: "white"
                                }
                                
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: {
                                        appController.executeCommand(model.name)
                                    }
                                    
                                    states: State {
                                        name: "pressed"
                                        when: parent.pressed
                                        PropertyChanges {
                                            target: parent.parent
                                            scale: 0.9
                                        }
                                    }
                                    
                                    transitions: Transition {
                                        NumberAnimation { properties: "scale"; duration: 100 }
                                    }
                                }
                            }
                            
                            // Кнопка редактировать / Edit button
                            Rectangle {
                                Layout.preferredWidth: 44
                                Layout.preferredHeight: 44
                                radius: 22
                                color: "#2196F3"
                                
                                Text {
                                    anchors.centerIn: parent
                                    text: "✎"
                                    font.pixelSize: 16
                                    color: "white"
                                }
                                
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: {
                                        console.log("Редактирование команды:", model.name)
                                        // TODO: открыть диалог редактирования
                                    }
                                    
                                    states: State {
                                        name: "pressed"
                                        when: parent.pressed
                                        PropertyChanges {
                                            target: parent.parent
                                            scale: 0.9
                                        }
                                    }
                                    
                                    transitions: Transition {
                                        NumberAnimation { properties: "scale"; duration: 100 }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Анимация при наведении / Hover animation
                    MouseArea {
                        id: mouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                    }
                    
                    // Анимация появления / Appearance animation
                    Component.onCompleted: {
                        opacity = 0
                        y = 20
                        appearAnimation.start()
                    }
                    
                    SequentialAnimation {
                        id: appearAnimation
                        ParallelAnimation {
                            NumberAnimation {
                                target: commandCard
                                property: "opacity"
                                to: 1.0
                                duration: 300
                                easing.type: Easing.OutCubic
                            }
                            NumberAnimation {
                                target: commandCard
                                property: "y"
                                to: 0
                                duration: 300
                                easing.type: Easing.OutCubic
                            }
                        }
                    }
                }
            }
        }
        
        // Информационная панель / Info panel
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 60
            radius: 12
            color: "#2a2a3e"
            border.color: Material.accent
            border.width: 1
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12
                
                Rectangle {
                    Layout.preferredWidth: 32
                    Layout.preferredHeight: 32
                    radius: 16
                    color: "#402196F3"
                    
                    Text {
                        anchors.centerIn: parent
                        text: "ℹ"
                        font.pixelSize: 18
                        color: Material.accent
                    }
                }
                
                Text {
                    Layout.fillWidth: true
                    text: qsTr("Всего команд: ") + commandsModel.count
                    font.pixelSize: 13
                    color: Material.foreground
                }
            }
        }
    }
    
    // Диалог добавления команды (заглушка) / Add command dialog (stub)
    Rectangle {
        id: addCommandDialog
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
                target: addCommandDialog
                property: "opacity"
                to: 1.0
                duration: 200
            }
        }
        
        SequentialAnimation {
            id: closeAnimation
            NumberAnimation {
                target: addCommandDialog
                property: "opacity"
                to: 0
                duration: 200
            }
            PropertyAction {
                target: addCommandDialog
                property: "visible"
                value: false
            }
        }
        
        MouseArea {
            anchors.fill: parent
            onClicked: addCommandDialog.close()
        }
        
        Rectangle {
            anchors.centerIn: parent
            width: 400
            height: 300
            radius: 20
            color: "#2a2a3e"
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 24
                spacing: 16
                
                Text {
                    text: qsTr("Добавить команду")
                    font.pixelSize: 20
                    font.bold: true
                    color: Material.foreground
                }
                
                Text {
                    text: qsTr("Функция в разработке")
                    font.pixelSize: 14
                    color: Material.color(Material.Grey)
                }
                
                Item {
                    Layout.fillHeight: true
                }
                
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    
                    Item {
                        Layout.fillWidth: true
                    }
                    
                    Rectangle {
                        Layout.preferredWidth: 100
                        Layout.preferredHeight: 40
                        radius: 20
                        color: Material.accent
                        
                        Text {
                            anchors.centerIn: parent
                            text: qsTr("Отмена")
                            font.pixelSize: 14
                            color: "white"
                        }
                        
                        MouseArea {
                            anchors.fill: parent
                            onClicked: addCommandDialog.close()
                        }
                    }
                }
            }
        }
    }
}
