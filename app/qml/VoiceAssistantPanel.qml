import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts

Item {
    id: root

    property string voiceInput: ""
    property string assistantResponse: ""

    ColumnLayout {
        anchors.fill: parent
        spacing: 14

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 96
            radius: 16
            color: "#2a2a3e"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 12

                Rectangle {
                    Layout.preferredWidth: 56
                    Layout.preferredHeight: 56
                    radius: 28
                    gradient: Gradient {
                        GradientStop { position: 0; color: appController.isVoiceAssistantActive ? "#00c853" : "#607d8b" }
                        GradientStop { position: 1; color: appController.isVoiceAssistantActive ? "#00acc1" : "#455a64" }
                    }
                    Text {
                        anchors.centerIn: parent
                        text: appController.isVoiceAssistantActive ? "🎙" : "🔇"
                        font.pixelSize: 24
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Text {
                        text: qsTr("Голосовой помощник")
                        font.pixelSize: 16
                        font.bold: true
                        color: "white"
                    }
                    Text {
                        text: appController.isVoiceAssistantActive
                              ? qsTr("Активен: можно говорить или отправить текстовую команду")
                              : qsTr("Не активен: нажмите Запустить")
                        color: "#CFD8DC"
                        font.pixelSize: 12
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 48
                radius: 12
                color: "#00695C"
                opacity: appController.isVoiceAssistantActive ? 0.5 : 1.0

                Text {
                    anchors.centerIn: parent
                    text: qsTr("▶ Запустить")
                    color: "white"
                    font.bold: true
                }

                MouseArea {
                    anchors.fill: parent
                    enabled: !appController.isVoiceAssistantActive
                    onClicked: {
                        appController.startVoiceAssistant("ru", true, true)
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 48
                radius: 12
                color: "#C62828"
                opacity: appController.isVoiceAssistantActive ? 1.0 : 0.5

                Text {
                    anchors.centerIn: parent
                    text: qsTr("■ Остановить")
                    color: "white"
                    font.bold: true
                }

                MouseArea {
                    anchors.fill: parent
                    enabled: appController.isVoiceAssistantActive
                    onClicked: appController.stopVoiceAssistant()
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 120
            radius: 16
            color: "#2a2a3e"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                Text {
                    text: qsTr("Понять как работает помощник")
                    font.pixelSize: 14
                    font.bold: true
                    color: "white"
                }

                Text {
                    text: qsTr("1) Запустите помощника. 2) Скажите wake-word 'ассистент' и команду. 3) Или введите текст ниже и нажмите Отправить.")
                    color: "#CFD8DC"
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 150
            radius: 16
            color: "#2a2a3e"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8

                TextField {
                    id: commandInput
                    Layout.fillWidth: true
                    placeholderText: qsTr("Например: привет, команды, статус, открыть браузер")
                    text: root.voiceInput
                    onTextChanged: root.voiceInput = text
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 42
                    radius: 10
                    color: Material.accent

                    Text {
                        anchors.centerIn: parent
                        text: qsTr("Отправить текстовую команду")
                        color: "white"
                        font.bold: true
                    }

                    MouseArea {
                        anchors.fill: parent
                        enabled: commandInput.text.length > 0
                        onClicked: {
                            root.assistantResponse = appController.processVoiceCommand(commandInput.text)
                        }
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: root.assistantResponse.length > 0 ? (qsTr("Ответ: ") + root.assistantResponse) : qsTr("Ответ появится здесь")
                    color: "#E1F5FE"
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                }
            }
        }
    }
}
