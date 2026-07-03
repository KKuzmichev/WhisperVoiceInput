"""Точка входа для консольной команды voiceinput."""

import os

os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::UserWarning:multiprocessing.resource_tracker",
)

import argparse
import sys
import threading

from .app import VoiceInputApp, CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Push-to-talk голосовой ввод через faster-whisper."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Размер модели faster-whisper: tiny, base, small, medium, large, turbo "
        "(по умолчанию из CONFIG). turbo — ускоренная large-v3-turbo.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Язык распознавания, напр. ru, en (по умолчанию из CONFIG). "
        "Поставьте 'auto' для автоопределения.",
    )
    parser.add_argument(
        "--no-menubar",
        action="store_true",
        help="Отключить индикатор в строке меню macOS (включён по умолчанию на macOS).",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Включить разделение спикеров (нужен extra: pip install -e .[diarize]). "
        "Если обнаружен один спикер — текст без меток; если несколько — с метками "
        "Speaker N: ...",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Мин. число спикеров для --diarize (по умолчанию 1).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Макс. число спикеров для --diarize (по умолчанию 4).",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Путь к аудиофайлу для транскрипции вместо записи с микрофона.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Путь сохранения расшифровки. Без --output сохраняется .txt рядом "
        "с аудиофайлом. Можно указать файл или директорию.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Печатать распознанный текст и диагностические сообщения в консоль.",
    )
    args = parser.parse_args()

    config = dict(CONFIG)
    if args.model is not None:
        config["model_size"] = args.model
    if args.language is not None:
        config["language"] = None if args.language == "auto" else args.language
    if args.diarize:
        config["diarize"] = True
    if args.min_speakers is not None:
        config["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        config["max_speakers"] = args.max_speakers
    if args.debug:
        config["debug"] = True

    app = VoiceInputApp(config)

    if args.file:
        app.transcribe_file(args.file, args.output)
        return

    use_menubar = sys.platform == "darwin" and not args.no_menubar

    if use_menubar:
        try:
            import rumps  # noqa: F401
        except ImportError:
            print(
                "rumps не установлен — menubar отключён. Установите: pip install rumps",
                file=sys.stderr,
            )
            use_menubar = False

    if use_menubar:
        from .menubar import MenubarIndicator

        indicator = MenubarIndicator(app)
        app.on_state_change = indicator.set_state

        worker = threading.Thread(target=app.run, daemon=True)
        worker.start()
        indicator.run()
    else:
        app.run()


if __name__ == "__main__":
    main()