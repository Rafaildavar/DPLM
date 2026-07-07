#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для запуска голосового помощника
Test script to launch voice assistant
"""

import sys
from pathlib import Path

# Добавить корневую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from app.services.voice_assistant import create_voice_assistant
    print("[✓] Модуль голосового помощника загружен")
except ImportError as e:
    print(f"[!] Ошибка импорта: {e}")
    sys.exit(1)


def main():
    """Главная функция запуска / Main launch function"""
    print("=" * 60)
    print("GestureBind Голосовой помощник / Voice Assistant")
    print("=" * 60)
    print()
    
    # Выбор языка / Language selection
    print("Выберите язык / Select language:")
    print("1. Русский (Russian)")
    print("2. English")
    print("3. Deutsch")
    
    choice = input("\nВаш выбор (1-3) / Your choice (1-3): ").strip()
    
    lang_map = {
        "1": "ru",
        "2": "en",
        "3": "de"
    }
    
    language = lang_map.get(choice, "ru")
    
    # Настройки / Settings
    print("\nНастройки / Settings:")
    wake_word = input("Включить активацию по ключевому слову? (y/n) / Enable wake word? (y/n): ").strip().lower() == 'y'
    tts = input("Включить озвучку? (y/n) / Enable TTS? (y/n): ").strip().lower() == 'y'
    
    print("\n" + "=" * 60)
    print("Создание голосового помощника... / Creating voice assistant...")
    print("=" * 60)
    
    # Создание помощника / Create assistant
    try:
        assistant = create_voice_assistant(
            language=language,
            wake_word=wake_word,
            tts=tts,
            stt_engine="auto"
        )
        
        if not assistant:
            print("[!] Не удалось создать голосового помощника")
            return
        
        print("[✓] Голосовой помощник создан успешно")
        print()
        
        # Информация / Information
        print("Информация / Information:")
        print(f"  Язык / Language: {language}")
        print(f"  Wake word: {'Включен' if wake_word else 'Выключен'} / {'Enabled' if wake_word else 'Disabled'}")
        print(f"  TTS: {'Включен' if tts else 'Выключен'} / {'Enabled' if tts else 'Disabled'}")
        print()
        
        if wake_word:
            wake_words_ru = ["ассистент", "помощник", "эй", "слушай"]
            wake_words_en = ["assistant", "hey", "listen", "okay"]
            wake_words = wake_words_ru if language == "ru" else wake_words_en
            print(f"Ключевые слова активации / Wake words: {', '.join(wake_words)}")
            print()
        
        print("Доступные команды / Available commands:")
        commands_ru = ["привет", "помощь", "статус", "команды", "спасибо", "пока"]
        commands_en = ["hello", "help", "status", "commands", "thanks", "goodbye"]
        commands = commands_ru if language == "ru" else commands_en
        print(f"  {', '.join(commands)}")
        print()
        
        print("=" * 60)
        print("Запуск прослушивания... / Starting listening...")
        print("Нажмите Ctrl+C для остановки / Press Ctrl+C to stop")
        print("=" * 60)
        print()
        
        # Запуск / Start
        assistant.start_listening_loop()
        
        # Ожидание / Wait
        try:
            import time
            while True:
                time.sleep(1)
                # Показываем статистику каждые 10 секунд / Show stats every 10 seconds
                stats = assistant.get_stats()
                if stats.get("wake_word_detected", 0) > 0 or stats.get("commands_executed", 0) > 0:
                    print(f"\r[Статистика] Команд: {stats.get('commands_executed', 0)}, "
                          f"Wake words: {stats.get('wake_word_detected', 0)}, "
                          f"Ошибок: {stats.get('errors', 0)}", end="")
        except KeyboardInterrupt:
            print("\n\n[!] Остановка голосового помощника...")
            assistant.stop_listening_loop()
            print("[✓] Голосовой помощник остановлен")
        
    except Exception as e:
        print(f"[!] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
