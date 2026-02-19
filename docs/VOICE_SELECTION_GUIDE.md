# Руководство по выбору голоса помощника

**Дата:** 12 февраля 2026  
**Версия:** 0.7.0

---

## Обзор

Голосовой помощник DPLM поддерживает выбор из множества доступных голосов системы. На macOS доступно более 180 голосов на разных языках.

---

## Выбор голоса через GUI

### Шаги:

1. **Откройте приложение DPLM**
2. **Перейдите в Настройки** (кнопка ⚙ в правом верхнем углу)
3. **В разделе "Голосовой помощник":**
   - Выберите язык помощника (Русский, English, Deutsch)
   - В выпадающем списке "Голос помощника" появится список доступных голосов для выбранного языка
   - Выберите нужный голос
   - Нажмите кнопку **▶** рядом со списком для предпросмотра голоса

### Настройка параметров голоса:

- **Скорость речи:** 50-300 WPM (слов в минуту)
  - По умолчанию: 150 WPM
  - Медленная речь: 50-100 WPM
  - Нормальная речь: 150-200 WPM
  - Быстрая речь: 250-300 WPM

- **Громкость:** 0-100%
  - По умолчанию: 90%
  - Тихая речь: 30-50%
  - Нормальная речь: 70-90%
  - Громкая речь: 90-100%

---

## Выбор голоса программно

### Получение списка голосов:

```python
from app.services.voice_assistant import create_voice_assistant

# Создать помощника
assistant = create_voice_assistant(language="ru", tts=True)

# Получить все доступные голоса
all_voices = assistant.get_available_voices()
print(f"Всего голосов: {len(all_voices)}")

# Получить голоса для конкретного языка
russian_voices = assistant.get_available_voices(language_filter="ru")
for voice in russian_voices:
    print(f"  - {voice['name']} ({voice['id']})")
```

### Установка голоса:

```python
# Установить голос по ID
assistant.set_voice("com.apple.speech.synthesis.voice.Yuri")

# Получить текущий голос
current_voice = assistant.get_current_voice()
print(f"Текущий голос: {current_voice['name']}")
```

### Предпросмотр голоса:

```python
# Предпросмотр с текстом по умолчанию
assistant.preview_voice()

# Предпросмотр с пользовательским текстом
assistant.preview_voice("Это тест выбранного голоса")
```

### Настройка параметров:

```python
# Установить скорость речи (50-300)
assistant.set_speech_rate(180)

# Установить громкость (0.0-1.0)
assistant.set_volume(0.8)
```

---

## Примеры использования

### Пример 1: Выбор русского голоса

```python
from app.services.voice_assistant import create_voice_assistant

assistant = create_voice_assistant(language="ru", tts=True)

# Получить русские голоса
ru_voices = assistant.get_available_voices(language_filter="ru")

# Выбрать первый русский голос
if ru_voices:
    assistant.set_voice(ru_voices[0]['id'])
    assistant.preview_voice("Привет, это тест голоса")
```

### Пример 2: Переключение между голосами

```python
assistant = create_voice_assistant(language="en", tts=True)

# Получить английские голоса
en_voices = assistant.get_available_voices(language_filter="en")

# Переключиться между голосами
for voice in en_voices[:3]:  # Первые 3 голоса
    print(f"Тестирую: {voice['name']}")
    assistant.set_voice(voice['id'])
    assistant.preview_voice(f"Hello, this is {voice['name']}")
```

### Пример 3: Настройка скорости и громкости

```python
assistant = create_voice_assistant(language="ru", tts=True)

# Медленная речь с низкой громкостью
assistant.set_speech_rate(80)
assistant.set_volume(0.5)
assistant.speak("Медленная тихая речь")

# Быстрая речь с высокой громкостью
assistant.set_speech_rate(250)
assistant.set_volume(1.0)
assistant.speak("Быстрая громкая речь")
```

---

## Интеграция с AppController (QML)

### Получение голосов в QML:

```qml
// Получить голоса для языка
var voices = appController.getAvailableVoicesByLanguage("ru")

// Установить голос
appController.setVoiceAssistantVoice(voiceId)

// Получить текущий голос
var currentVoice = appController.getCurrentVoice()

// Предпросмотр голоса
appController.previewVoiceAssistantVoice("Тест голоса")

// Настройка параметров
appController.setVoiceAssistantSpeechRate(180)
appController.setVoiceAssistantVolume(0.9)
```

---

## Определение языка голоса

Система автоматически определяет язык голоса по его имени и ID:

- **Русский:** содержит "russian", "ru", "русск"
- **Английский:** содержит "english", "en", "eng", "англ"
- **Немецкий:** содержит "german", "de", "deutsch", "немец"

Если язык не определен, голос помечается как "unknown" и доступен для всех языков.

---

## Особенности платформ

### macOS:

- Доступно более 180 голосов
- Голоса включают компактные (compact) и расширенные версии
- Поддержка множества языков и акцентов
- Голоса имеют формат ID: `com.apple.speech.synthesis.voice.NAME`

### Windows:

- Меньше голосов по умолчанию
- Может потребоваться установка дополнительных языковых пакетов
- Голоса имеют формат ID: `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\NAME`

### Linux:

- Зависит от установленных TTS движков (espeak, festival)
- Может потребоваться установка дополнительных голосов

---

## Рекомендации

1. **Для русского языка:** Выберите голос с "russian" или "ru" в названии
2. **Для английского языка:** Выберите голос с "english" или "en" в названии
3. **Скорость речи:** 150-200 WPM оптимальна для большинства пользователей
4. **Громкость:** 80-90% обеспечивает хорошую слышимость без дискомфорта
5. **Предпросмотр:** Всегда используйте предпросмотр перед окончательным выбором

---

## Устранение неполадок

### Проблема: Список голосов пуст

**Решение:**
1. Убедитесь, что TTS включен (`tts_enabled=True`)
2. Проверьте установку pyttsx3: `pip install pyttsx3`
3. На macOS проверьте доступность голосов через системные настройки

### Проблема: Голос не меняется

**Решение:**
1. Убедитесь, что голосовой помощник активен
2. Проверьте правильность ID голоса
3. Попробуйте перезапустить помощника

### Проблема: Голос говорит на неправильном языке

**Решение:**
1. Выберите голос, соответствующий языку помощника
2. Проверьте фильтр языка при получении списка голосов
3. Убедитесь, что язык помощника совпадает с языком голоса

---

*Документ обновлен: 12 февраля 2026*
