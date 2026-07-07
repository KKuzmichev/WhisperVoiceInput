# WhisperVoiceInput

Push-to-talk голосовой ввод с локальной транскрипцией через faster-whisper. Распознанный текст печатается в текущее активное окно как будто введён с клавиатуры.

## Возможности
- Глобальная горячая клавиша (работает, когда приложение не в фокусе)
- Локальный вывод на CPU/GPU через faster-whisper (без облака, без API-ключей)
- Эмуляция ввода клавиатуры в любое активное окно
- Кроссплатформенно: macOS / Linux / Windows
- Опциональное разделение спикеров (диаризация) через `speechbrain` ECAPA-TDNN
- Транскрипция аудиофайла в `.txt` (без записи с микрофона)
- Транскрипция видеофайла в `.txt` с таймлайнами (через ffmpeg)

## Установка

### Из исходников (устанавливаемый пакет)
```bash
pip install .
# или в режиме разработки:
pip install -e .
```
После установки доступна консольная команда `voiceinput`.

### С разделением спикеров (диаризация)
```bash
pip install -e .[diarize]
```
Этот extra тянет `speechbrain` + `torch` (~500–700 МБ). Модель ECAPA-TDNN
(`speechbrain/spkrec-ecapa-voxceleb`) скачивается при первом запуске `--diarize`
с HuggingFace **без токена** (открытая лицензия) и кешируется в `~/.cache/huggingface/`.

#### Как работает диаризация
1. **faster-whisper** транскрибирует аудио с включённым VAD-фильтром — каждый
   сегмент получает таймстампы `(start, end)` и текст.
2. Для каждого сегмента вырезается соответствующий фрагмент аудио и пропускается
   через модель **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`,
   обучена на VoxCeleb) — получается 192-мерный speaker embedding.
3. Embeddings кластеризуются агломеративным методом (`sklearn.AgglomerativeClustering`,
   cosine metric, average linkage) с подбором числа кластеров в диапазоне
   `[min_speakers, max_speakers]` — каждый кластер = один спикер.
4. Если обнаружен **один спикер** — текст выводится без меток (как обычно).
   Если **несколько** — каждый сегмент помечается `Speaker N: ...`.

Модель ECAPA-TDNN (~80 МБ) кешируется в `~/.cache/whisper-voice-input/spkrec-ecapa-voxceleb/`.

### С транскрипцией видео
```bash
pip install -e .[video]
```
Этот extra тянет `imageio-ffmpeg` (~80 МБ), который включает бинарник `ffmpeg` для извлечения аудиодорожки из видеофайла.

### В виртуальном окружении
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install .
```

Для транскрипции файлов дополнительно нужен `soundfile` (уже входит в зависимости):
```bash
pip install soundfile
```
Если частота аудио отличается от 16000 Гц, дополнительно потребуется `librosa`:
```bash
pip install librosa
```

При первом запуске faster-whisper скачает веса модели (по умолчанию `base`, ~140 МБ) и закеширует их.

## Хранение моделей
Модели кешируются в директории HuggingFace Hub:
```
~/.cache/huggingface/hub/
```
Например, модель `base` сохраняется в:
```
~/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/<хэш>/
```

- Чтобы изменить место хранения, задайте переменную окружения `HF_HOME`:
  ```bash
  export HF_HOME=/path/to/custom/cache
  voiceinput
  ```
- Чтобы очистить кеш моделей:
  ```bash
  rm -rf ~/.cache/huggingface/hub/models--*whisper*
  ```

## Разрешения
- **macOS**
  - System Settings → Privacy & Security → **Accessibility**: добавить терминал / Python, в котором запускается скрипт (нужно для эмуляции ввода).
  - System Settings → Privacy & Security → **Microphone**: добавить терминал / Python (нужно для записи звука).
- **Linux**
  - pynput работает с **X11**. На Wayland эмуляция ввода клавиатуры может не работать.
  - Для некоторых окружений может потребоваться `xdotool`.
- **Windows**
  - **Микрофон**: Параметры → Конфиденциальность → Микрофон → включить «Доступ к микрофону для приложений» и «Разрешить приложениям доступ к микрофону». Для voiceinput (Python/PowerShell) нужно включить доступ в секции **«Для классических приложений»**, а не «Для приложений Microsoft Store».
  - **Эмуляция ввода**: специальных разрешений не требуется.

## Конфигурация
Отредактируйте словарь `CONFIG` в `src/whisper_voice_input/app.py`:
- `toggle_key` — клавиша переключения записи, по умолчанию `ctrl`.
- `double_press_interval` — интервал двойного нажатия в секундах, по умолчанию `0.35`.
- `model_size` — размер модели: `tiny`, `base`, `small`, `medium`, `large`, `turbo`. По умолчанию `base`.
- `language` — язык распознавания, по умолчанию `ru`. Поставьте `None` для автоопределения.
- `sample_rate` — частота дискретизации записи, по умолчанию `16000`.
- `device_index` — индекс устройства ввода (`None` = системное по умолчанию).
- `diarize` — включить разделение спикеров, по умолчанию `False`.
- `min_speakers` / `max_speakers` — границы числа спикеров для диаризации (1 и 4).

## Удалённый запуск (SSH + PulseAudio)
Запускайте voiceinput на удалённом сервере, используя микрофон клиента через SSH-туннель.

**Клиент (macOS):**
```bash
brew install pulseaudio
# Запуск PulseAudio с TCP-поддержкой:
pulseaudio --load="module-native-protocol-tcp auth-ip-acl=127.0.0.1" --exit-idle-time=-1
# Подключение к серверу с пробросом PulseAudio:
ssh -R 4713:localhost:4713 user@server
```

**Сервер (Linux):**
```bash
sudo apt install pulseaudio    # PulseAudio клиент
# Проверить микрофон клиента:
PULSE_SERVER=tcp:localhost:4713 pactl list sources short
# Запустить voiceinput:
PULSE_SERVER=tcp:localhost:4713 voiceinput --model=base --language=ru
```
PortAudio автоматически подхватывает `PULSE_SERVER` — код менять не нужно.

## Запуск
```bash
voiceinput
```
Для выхода нажмите `Ctrl+C` в терминале.

### Параметры запуска
```bash
voiceinput --model small --language ru
```
- `--model` — размер модели: `tiny`, `base`, `small`, `medium`, `large`, `turbo` (по умолчанию `base`). `turbo` — ускоренная `large-v3-turbo`, почти не уступает `large` по точности, но работает заметно быстрее.
- `--language` — язык распознавания, напр. `ru`, `en`. Поставьте `auto` для автоопределения (по умолчанию `ru`).
- `--no-menubar` — (только macOS) отключить индикатор в строке меню. По умолчанию на macOS menubar включён автоматически. Иконка-приведение: белое — ожидание, чёрное — запись.
- `--diarize` — включить разделение спикеров. Если обнаружен один спикер — текст выводится без меток (как обычно). Если несколько — каждый сегмент помечается `Speaker N: ...`. Нужен extra `diarize` (см. ниже).
- `--min-speakers` / `--max-speakers` — ограничения числа спикеров для `--diarize` (по умолчанию 1 и 4).
- `--file PATH` — путь к аудиофайлу для транскрипции вместо записи с микрофона. Поддерживаемые форматы зависят от `soundfile` (wav, flac, ogg, mp3 и др.).
- `--video PATH` — путь к видеофайлу для транскрипции (нужен extra `video`). Аудиодорожка извлекается через ffmpeg, далее транскрибируется faster-whisper.
- `--output PATH` — путь сохранения расшифровки. Без `--output` `.txt` сохраняется рядом с файлом с тем же именем. Можно указать файл или директорию (в последнем случае `.txt` с именем файла сохранится в неё).
- `--timeline` — добавить таймлайны к расшифровке `--file` (формат `[HH:MM:SS,mmm] текст`).
- `--no-timeline` — отключить таймлайны для `--video` (по умолчанию для видео таймлайны включены, для аудио выключены).
- `--debug` — печатать распознанный текст и диагностику в консоль.

Размеры моделей (приблизительно):
| Модель | Размер | Точность | Скорость |
|--------|--------|----------|----------|
| `tiny` | ~75 МБ | низкая | очень быстро |
| `base` | ~140 МБ | средняя | быстро |
| `small` | ~460 МБ | хорошая | средне |
| `medium`| ~1.5 ГБ | высокая | медленно |
| `large` | ~3 ГБ | лучшая | медленно |
| `turbo` | ~1.5 ГБ | близка к large | заметно быстрее large |

## Использование
1. Запустите `voiceinput`, дождитесь сообщения о загрузке модели.
2. Перейдите в любое окно с полем ввода (текстовый редактор, браузер, мессенджер).
3. Дважды нажмите `Ctrl` — начнётся запись (индикатор `● ЗАПИСЬ...` в терминале).
4. Произнесите фразу.
5. Дважды нажмите `Ctrl` — запись остановится, распознанный текст напечатается в активное окно.

## Транскрипция файла
```bash
# Сохранит рядом с аудиофайлом: recording.wav -> recording.txt
voiceinput --file recording.wav

# Указать путь к .txt
voiceinput --file recording.wav --output transcript.txt

# Указать директорию — .txt с именем аудио сохранится в неё
voiceinput --file recording.wav --output /path/to/dir/

# Транскрипция файла с разделением спикеров
voiceinput --file meeting.wav --diarize --output meeting.txt

# С таймлайнами
voiceinput --file recording.wav --timeline
```
Без `--output` расшифровка сохраняется рядом с аудиофайлом с тем же именем и расширением `.txt`. Если `--output` указывает на директорию — `.txt` с именем аудио сохранится в неё; если путь без расширения — добавляется `.txt`.

## Транскрипция видео
```bash
# Сохранит clip.mp4 -> clip.txt с таймлайнами (по умолчанию)
voiceinput --video clip.mp4

# Без таймлайнов
voiceinput --video clip.mp4 --no-timeline

# Указать путь сохранения
voiceinput --video clip.mp4 --output /path/to/subtitles.txt

# Видео с разделением спикеров + таймлайны
voiceinput --video interview.mp4 --diarize --output interview.txt
```
Формат с таймлайнами: `[HH:MM:SS,mmm] текст` (или `Speaker N: текст` при диаризации). Аудиодорожка извлекается через встроенный ffmpeg (`imageio-ffmpeg`), затем транскрибируется faster-whisper.
