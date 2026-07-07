# Руководство по голосовому помощнику GestureBind

**Дата:** 12 февраля 2026  
**Версия:** 0.7.0

---

## 📋 Содержание

1. [Обзор](#обзор)
2. [Возможности](#возможности)
3. [Установка и настройка](#установка-и-настройка)
4. [Использование](#использование)
5. [Команды](#команды)
6. [Интеграция](#интеграция)
7. [Устранение неполадок](#устранение-неполадок)

---

## Обзор

Голосовой помощник GestureBind — полнофункциональный ассистент с базовым функционалом современных голосовых помощников (Siri, Google Assistant, Alexa).

### Основные компоненты:

- **STT (Speech-to-Text)**: Распознавание речи через SpeechRecognition (Google) и Vosk (офлайн)
- **TTS (Text-to-Speech)**: Синтез речи через pyttsx3
- **NLU (Natural Language Understanding)**: Понимание намерений пользователя
- **Диалоговая система**: Контекстные ответы и поддержание разговора
- **Wake Word Detection**: Активация по ключевому слову
- **Выполнение команд**: Интеграция с системой команд GestureBind

---

## Возможности

### ✅ Реализовано:

1. **Распознавание речи (STT)**
   - Google Speech Recognition (требует интернет)
   - Vosk (офлайн, требует модель)
   - CMU Sphinx (офлайн, базовое качество)
   - Поддержка русского, английского, немецкого

2. **Синтез речи (TTS)**
   - pyttsx3 (кроссплатформенный)
   - Настройка скорости и громкости
   - Выбор голоса по языку

3. **Диалоговая система**
   - Обработка приветствий
   - Контекстные ответы
   - Сохранение истории разговора
   - Обработка неизвестных команд

4. **Wake Word Detection**
   - Ключевые слова: "ассистент", "помощник", "эй" (RU)
   - Ключевые слова: "assistant", "hey", "listen" (EN)
   - Опциональная активация

5. **Выполнение команд**
   - Интеграция с CommandExecutor
   - Голосовое управление системой
   - Выполнение пользовательских команд

6. **Статистика и мониторинг**
   - Отслеживание выполненных команд
   - Подсчет ошибок
   - Состояние помощника

---

## Установка и настройка

### Требования:

```bash
# Основные библиотеки
pip install SpeechRecognition pyttsx3

# Опционально: Vosk для офлайн распознавания
pip install vosk

# Опционально: CMU Sphinx для офлайн STT
pip install pocketsphinx
```

### Настройка микрофона:

1. **macOS**: Проверьте разрешения в `System Preferences → Security & Privacy → Microphone`
2. **Windows**: Проверьте настройки микрофона в `Settings → Privacy → Microphone`

### Настройка Vosk (опционально):

1. Скачайте модель Vosk с [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
2. Распакуйте в `models/vosk/`
3. Используйте `stt_engine="vosk"` при создании помощника

---

## Использование

### Базовое использование:

```python
from app.services.voice_assistant import create_voice_assistant

# Создание помощника
assistant = create_voice_assistant(
    language="ru",      # "ru", "en", "de"
    wake_word=True,     # Активация по ключевому слову
    tts=True,          # Включить синтез речи
    stt_engine="auto"  # "auto", "google", "vosk", "sphinx"
)

# Запуск цикла прослушивания
assistant.start_listening_loop()

# Остановка
assistant.stop_listening_loop()
```

### Использование в GUI:

1. Откройте приложение GestureBind
2. Перейдите в **Настройки** (⚙)
3. В разделе **"Голосовой помощник"**:
   - Выберите язык
   - Настройте скорость речи
   - Включите/выключите wake word
   - Нажмите **"▶ Запустить"**

### Программное использование:

```python
from app.services.voice_assistant import VoiceAssistant, Language

# Создание с полными параметрами
assistant = VoiceAssistant(
    language=Language.RUSSIAN,
    wake_word_enabled=True,
    tts_enabled=True,
    stt_engine="google"
)

# Регистрация пользовательской команды
def my_command_handler(text: str) -> str:
    return "Команда выполнена!"

assistant.register_command("моя команда", my_command_handler)

# Обработка команды напрямую
response = assistant.process_command("привет")
assistant.speak(response)

# Запуск автоматического прослушивания
assistant.start_listening_loop()
```

---

## Команды

### Системные команды:

| Команда (RU) | Команда (EN) | Описание |
|--------------|--------------|----------|
| `привет` | `hello` | Приветствие |
| `помощь` | `help` | Список возможностей |
| `спасибо` | `thanks` | Благодарность |
| `пока` | `goodbye` | Прощание |
| `стоп` | `stop` | Остановить помощника |
| `выход` | `exit` | Выход |
| `статус` | `status` | Статус системы |
| `команды` | `commands` | Список команд |

### Команды управления системой:

| Команда (RU) | Команда (EN) | Действие |
|--------------|--------------|----------|
| `открыть браузер` | `open browser` | Открыть браузер |
| `увеличить громкость` | `volume up` | Увеличить громкость |
| `уменьшить громкость` | `volume down` | Уменьшить громкость |

### Регистрация пользовательских команд:

```python
# Регистрация команды с точным совпадением
assistant.register_command("открыть калькулятор", lambda text: "Открываю калькулятор")

# Регистрация команды с regex паттерном
import re
assistant.register_command(r"открыть (.+)", lambda text: f"Открываю {re.search(r'открыть (.+)', text).group(1)}")
```

---

## Интеграция

### Интеграция с AppController:

```python
# В app/main.py
from app.services.voice_assistant import create_voice_assistant

class AppController(QObject):
    @Slot(str, bool, bool, result=bool)
    def startVoiceAssistant(self, language: str, wake_word: bool, tts: bool):
        assistant = create_voice_assistant(language, wake_word, tts)
        if assistant:
            assistant.start_listening_loop()
            return True
        return False
```

### Интеграция с CommandExecutor:

```python
from app.services.command_executor import get_executor

# Регистрация команд из CommandExecutor в голосовом помощнике
executor = get_executor()
assistant = create_voice_assistant()

for cmd_name in executor.commands_registry.keys():
    assistant.register_command(
        cmd_name,
        lambda text, name=cmd_name: executor.execute(name)
    )
```

### Интеграция с жестами:

```python
# Выполнение команды по жесту через голосовой помощник
def on_gesture_detected(gesture_label: str):
    # Поиск команды по жесту
    command = find_command_by_gesture(gesture_label)
    if command:
        assistant.speak(f"Выполняю команду: {command.name}")
        executor.execute(command.name)
```

---

## Устранение неполадок

### Проблема: Микрофон не работает

**Решение:**
1. Проверьте разрешения системы на доступ к микрофону
2. Убедитесь, что микрофон подключен и работает
3. Проверьте уровень громкости микрофона

```python
# Тест микрофона
import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    print("Говорите...")
    audio = r.listen(source, timeout=5)
    print("Услышал!")
```

### Проблема: Низкое качество распознавания

**Решение:**
1. Используйте внешний микрофон
2. Уменьшите фоновый шум
3. Говорите четко и медленно
4. Попробуйте другой STT движок (Vosk для офлайн)

```python
# Использование Vosk для лучшего качества
assistant = create_voice_assistant(stt_engine="vosk")
```

### Проблема: TTS не работает

**Решение:**
1. Проверьте установку pyttsx3: `pip install pyttsx3`
2. На macOS может потребоваться установка дополнительных голосов
3. Проверьте настройки системы для синтеза речи

```python
# Тест TTS
import pyttsx3
engine = pyttsx3.init()
engine.say("Тест")
engine.runAndWait()
```

### Проблема: Wake word не срабатывает

**Решение:**
1. Убедитесь, что wake word включен: `wake_word_enabled=True`
2. Произносите ключевое слово четко
3. Проверьте список ключевых слов в коде

```python
# Отключение wake word для постоянной активации
assistant = create_voice_assistant(wake_word=False)
```

---

## API Reference

### VoiceAssistant

```python
class VoiceAssistant:
    def __init__(
        self,
        language: Language = Language.RUSSIAN,
        wake_word_enabled: bool = True,
        tts_enabled: bool = True,
        stt_engine: str = "auto"
    )
    
    def speak(text: str, async_mode: bool = False)
    def listen(timeout: float = 5.0, phrase_time_limit: float = 5.0) -> Optional[str]
    def process_command(text: str) -> Optional[str]
    def register_command(pattern: str, handler: Callable)
    def start_listening_loop()
    def stop_listening_loop()
    def get_stats() -> Dict[str, Any]
```

### create_voice_assistant()

```python
def create_voice_assistant(
    language: str = "ru",
    wake_word: bool = True,
    tts: bool = True,
    stt_engine: str = "auto"
) -> Optional[VoiceAssistant]
```

---

## Примеры использования

### Пример 1: Простой помощник

```python
from app.services.voice_assistant import create_voice_assistant

assistant = create_voice_assistant(language="ru", wake_word=True)
assistant.start_listening_loop()

# Говорите: "ассистент привет"
# Ответ: "Привет! Чем могу помочь?"
```

### Пример 2: Помощник с пользовательскими командами

```python
assistant = create_voice_assistant(language="ru")

def open_app_handler(text: str) -> str:
    app_name = text.split()[-1]  # Последнее слово
    # Логика открытия приложения
    return f"Открываю {app_name}"

assistant.register_command("открыть", open_app_handler)
assistant.start_listening_loop()
```

### Пример 3: Интеграция с GUI

```python
# В QML
Button {
    text: "Запустить помощника"
    onClicked: {
        appController.startVoiceAssistant("ru", true, true)
    }
}

// Отображение статуса
Text {
    text: appController.isVoiceAssistantActive ? "Активен" : "Неактивен"
}
```

---

## Дальнейшее развитие

### Планируемые функции:

- [ ] Поддержка wake word через ML модель (Porcupine, Snowboy)
- [ ] Интеграция с ChatGPT/LLM для умных ответов
- [ ] Поддержка многоязычного диалога
- [ ] Голосовые профили пользователей
- [ ] Обучение на пользовательских данных
- [ ] Интеграция с календарем и напоминаниями
- [ ] Управление умным домом

---

## Поддержка

При возникновении проблем:

1. Проверьте логи приложения
2. Убедитесь, что все зависимости установлены
3. Проверьте разрешения системы
4. Создайте issue в репозитории проекта

---

*Документ обновлен: 12 февраля 2026*
