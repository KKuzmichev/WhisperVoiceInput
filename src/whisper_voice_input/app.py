#!/usr/bin/env python3
"""WhisperVoiceInput — push-to-talk голосовой ввод через faster-whisper."""

import os
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from pynput import keyboard

from .diarize import Diarizer


CONFIG = {
    "toggle_key": "ctrl",
    "double_press_interval": 0.35,
    "model_size": "base",
    "language": "ru",
    "sample_rate": 16000,
    "device_index": None,
    "diarize": False,
    "min_speakers": 1,
    "max_speakers": 4,
}


class VoiceInputApp:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.typer = keyboard.Controller()

        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self.pressed_keys = set()
        self.last_ctrl_time = 0.0

        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stop_event = threading.Event()
        self.on_state_change = None
        self.diarizer = None

    MODEL_ALIASES = {
        "tiny": "Systran/faster-whisper-tiny",
        "base": "Systran/faster-whisper-base",
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large": "Systran/faster-whisper-large-v3",
        "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    }

    @staticmethod
    def _spinner(stop_event, message="загрузка модели..."):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not stop_event.is_set():
            sys.stdout.write("\r" + frames[i % len(frames)] + " " + message)
            sys.stdout.flush()
            i += 1
            stop_event.wait(0.1)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def load_model(self):
        model_name = self.config["model_size"]
        model_id = self.MODEL_ALIASES.get(model_name, model_name)
        print(f"Модель faster-whisper: {model_name} ({model_id})")

        try:
            model_path = snapshot_download(repo_id=model_id)
        except Exception as e:
            print(f"Ошибка скачивания модели: {e}", file=sys.stderr)
            print(
                "Убедитесь, что есть доступ к интернету для скачивания весов "
                "при первом запуске.",
                file=sys.stderr,
            )
            sys.exit(1)

        spinner_stop = threading.Event()
        spinner_thread = threading.Thread(
            target=self._spinner,
            args=(spinner_stop, "инициализация модели..."),
            daemon=True,
        )
        spinner_thread.start()
        try:
            self.model = WhisperModel(
                model_path, device="cpu", compute_type="int8"
            )
        except Exception as e:
            spinner_stop.set()
            spinner_thread.join()
            print(f"\nОшибка загрузки модели: {e}", file=sys.stderr)
            sys.exit(1)
        spinner_stop.set()
        spinner_thread.join()
        print("Модель загружена. Готово к работе.")
        if self.config.get("diarize"):
            self._load_diarizer()
        print("Двойное нажатие Ctrl — начать/остановить запись. Для выхода Ctrl+C.")
        if sys.platform == "darwin":
            print(
                "ВНИМАНИЕ: для перехвата клавиш на macOS добавьте терминал/Python в\n"
                "  System Settings → Privacy & Security → Accessibility.",
                file=sys.stderr,
            )

    def _load_diarizer(self):
        print("Загрузка модели диаризации (speechbrain ECAPA-TDNN)...")
        self.diarizer = Diarizer(
            min_speakers=self.config.get("min_speakers", 1),
            max_speakers=self.config.get("max_speakers", 4),
            sample_rate=self.config["sample_rate"],
        )
        ok = self.diarizer.load()
        if ok:
            print("Модель диаризации загружена.")
        else:
            self.diarizer = None
            print("Диаризация отключена. Продолжаю без разделения спикеров.",
                  file=sys.stderr)

    @staticmethod
    def _load_audio_file(path):
        import subprocess
        import tempfile
        from pathlib import Path

        try:
            import soundfile as sf
        except ImportError:
            sf = None

        if sf is not None:
            try:
                audio, sr = sf.read(str(path), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    try:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    except ImportError:
                        print(
                            f"Частота аудио {sr} Гц, нужно 16000. "
                            "Установите librosa: pip install librosa",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                return audio
            except Exception:
                pass

        try:
            import imageio_ffmpeg
        except ImportError:
            print(
                "Формат файла не поддерживается soundfile. "
                "Установите extra: pip install -e .[video]",
                file=sys.stderr,
            )
            sys.exit(1)

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_wav = Path(tempfile.mkdtemp()) / "audio.wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1",
             "-ar", "16000", "-f", "wav", str(tmp_wav)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            if sf is not None:
                audio, _ = sf.read(str(tmp_wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return audio
            import wave
            with wave.open(str(tmp_wav), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
        finally:
            tmp_wav.unlink(missing_ok=True)
            try:
                tmp_wav.parent.rmdir()
            except OSError:
                pass

    @staticmethod
    def _format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _segments_to_text(self, segments, with_timeline, tagged=None):
        if tagged is not None:
            if with_timeline:
                lines = []
                for (speaker, text), seg in zip(tagged, segments):
                    if text:
                        lines.append(
                            f"[{self._format_timestamp(seg.start)}] "
                            f"Speaker {speaker + 1}: {text.strip()}"
                        )
                return "\n".join(lines)
            n_speakers = self.diarizer.num_speakers(tagged) if tagged else 0
            if n_speakers <= 1:
                return "".join(t for _, t in tagged).strip()
            grouped = self._merge_adjacent_speakers(tagged)
            return "\n".join(f"Speaker {s + 1}: {t}" for s, t in grouped)
        if with_timeline:
            lines = []
            for seg in segments:
                if seg.text.strip():
                    lines.append(
                        f"[{self._format_timestamp(seg.start)}] {seg.text.strip()}"
                    )
            return "\n".join(lines)
        return "".join(seg.text for seg in segments).strip()

    def transcribe_file(self, input_path, output_path=None, with_timeline=False):
        input_path = Path(input_path).expanduser().resolve()
        if not input_path.is_file():
            print(f"Файл не найден: {input_path}", file=sys.stderr)
            sys.exit(1)

        if self.model is None:
            self.load_model()
        if self.config.get("diarize") and self.diarizer is None:
            self._load_diarizer()

        print(f"Загрузка аудио: {input_path}")
        audio = self._load_audio_file(input_path)
        duration = len(audio) / self.config["sample_rate"]
        print(f"Длительность: {duration:.1f}с")

        use_diarize = self.diarizer is not None
        print("Транскрипция началась...")
        segments, info = self.model.transcribe(
            audio,
            language=self.config["language"],
            beam_size=5,
            vad_filter=use_diarize,
            log_progress=True,
        )
        segments = list(segments)

        tagged = None
        if use_diarize and segments:
            tagged = self.diarizer.diarize(audio, segments)

        text = self._segments_to_text(segments, with_timeline, tagged)

        if output_path is None:
            output_path = input_path.with_suffix(".txt")
        else:
            output_path = Path(output_path).expanduser()
            if output_path.is_dir():
                output_path = output_path / input_path.with_suffix(".txt").name
            elif not output_path.suffix:
                output_path = output_path.with_suffix(".txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(text, encoding="utf-8")
        if self.config.get("debug"):
            print(f"Распознано ({len(segments)} сегм., {info.duration:.1f}с):")
            print(text)
        print(f"Сохранено в: {output_path}")

    def transcribe_video(self, input_path, output_path=None, with_timeline=True):
        import tempfile

        try:
            import imageio_ffmpeg
        except ImportError:
            print(
                "Для транскрипции видео нужен imageio-ffmpeg: "
                "pip install -e .[video]",
                file=sys.stderr,
            )
            sys.exit(1)

        input_path = Path(input_path).expanduser().resolve()
        if not input_path.is_file():
            print(f"Файл не найден: {input_path}", file=sys.stderr)
            sys.exit(1)

        if self.model is None:
            self.load_model()
        if self.config.get("diarize") and self.diarizer is None:
            self._load_diarizer()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Извлечение аудио из видео: {input_path}")
        tmp_wav = Path(tempfile.mkdtemp()) / "audio.wav"
        import subprocess

        subprocess.run(
            [ffmpeg, "-y", "-i", str(input_path), "-vn", "-ac", "1",
             "-ar", "16000", "-f", "wav", str(tmp_wav)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            audio = self._load_audio_file(tmp_wav)
        finally:
            tmp_wav.unlink(missing_ok=True)

        use_diarize = self.diarizer is not None
        print("Транскрипция началась...")
        segments, info = self.model.transcribe(
            audio,
            language=self.config["language"],
            beam_size=5,
            vad_filter=use_diarize,
            log_progress=True,
        )
        segments = list(segments)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        tagged = None
        if use_diarize and segments:
            tagged = self.diarizer.diarize(audio, segments)

        text = self._segments_to_text(segments, with_timeline, tagged)

        if output_path is None:
            output_path = input_path.with_suffix(".txt")
        else:
            output_path = Path(output_path).expanduser()
            if output_path.is_dir():
                output_path = output_path / input_path.with_suffix(".txt").name
            elif not output_path.suffix:
                output_path = output_path.with_suffix(".txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(text, encoding="utf-8")
        if self.config.get("debug"):
            print(f"Распознано ({len(segments)} сегм., {info.duration:.1f}с):")
            print(text)
        print(f"Сохранено в: {output_path}")

    def _set_indicator(self, text):
        sys.stdout.write("\r\033[K[" + text + "]\n")
        sys.stdout.flush()
        if self.on_state_change:
            if "ЗАПИСЬ" in text:
                self.on_state_change("recording")
            elif "СТОП" in text:
                self.on_state_change("processing")

    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def start_recording(self):
        with self.lock:
            if self.recording:
                return
            self.audio_chunks = []
            self.recording = True
        try:
            self.stream = sd.InputStream(
                samplerate=self.config["sample_rate"],
                channels=1,
                dtype="float32",
                device=self.config["device_index"],
                callback=self._audio_callback,
            )
            self.stream.start()
            self._set_indicator("● ЗАПИСЬ...")
        except Exception as e:
            print(f"Ошибка доступа к микрофону: {e}", file=sys.stderr)
            if "PortAudioError" in str(type(e).__name__) or isinstance(
                e, sd.PortAudioError
            ):
                self._print_mic_permission_hint()
            with self.lock:
                self.recording = False
            self.stream = None

    @staticmethod
    def _print_mic_permission_hint():
        print(
            "Нет доступа к микрофону. Проверьте разрешения:\n"
            "  macOS: System Settings → Privacy & Security → Microphone\n"
            "  Windows: Параметры → Конфиденциальность → Микрофон\n"
            "  Linux: проверьте, что устройство захвата доступно (arecord -l).",
            file=sys.stderr,
        )

    def stop_recording(self):
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            chunks = self.audio_chunks
            self.audio_chunks = []
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if not chunks:
            if self.config.get("debug"):
                print("Предупреждение: ничего не записано.", file=sys.stderr)
            return
        self._set_indicator("● СТОП")
        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        duration = len(audio) / self.config["sample_rate"]
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if self.config.get("debug"):
            print(f"Записано {duration:.2f}с аудио (peak={peak:.4f}).", file=sys.stderr)
        if peak < 0.01:
            print(
                "Аудио тихое/пустое. Проверьте разрешение на микрофон:\n"
                "  macOS: System Settings → Privacy & Security → Microphone "
                "(добавьте терминал/Python), затем перезапустите программу.",
                file=sys.stderr,
            )
            return
        self.executor.submit(self._transcribe_and_type, audio)

    def _transcribe_and_type(self, audio):
        try:
            use_diarize = self.diarizer is not None
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                segments, info = self.model.transcribe(
                    audio,
                    language=self.config["language"],
                    beam_size=5,
                    vad_filter=use_diarize,
                )
                segments = list(segments)

            if use_diarize and segments:
                tagged = self.diarizer.diarize(audio, segments)
                n_speakers = self.diarizer.num_speakers(tagged) if tagged else 0
                if n_speakers <= 1:
                    text = "".join(t for _, t in tagged).strip()
                else:
                    grouped = self._merge_adjacent_speakers(tagged)
                    text = "\n".join(
                        f"Speaker {s + 1}: {t}" for s, t in grouped
                    )
            else:
                text = "".join(seg.text for seg in segments).strip()

            if not text:
                if self.config.get("debug"):
                    print("Транскрипция пуста.", file=sys.stderr)
                if self.on_state_change:
                    self.on_state_change("idle")
                return
            if self.config.get("debug"):
                print(f"Распознано: {text}")
            try:
                self.typer.type(text)
            except Exception as e:
                print(
                    f"Ошибка эмуляции ввода: {e}\n"
                    "Возможно, не выдано разрешение на Accessibility "
                    "(macOS: System Settings → Privacy & Security → Accessibility).",
                    file=sys.stderr,
                )
            if self.on_state_change:
                self.on_state_change("idle")
        except Exception:
            print("Ошибка транскрипции:", file=sys.stderr)
            traceback.print_exc()

    @staticmethod
    def _merge_adjacent_speakers(tagged):
        merged = []
        cur_speaker, cur_text = None, ""
        for speaker, text in tagged:
            if cur_speaker is None:
                cur_speaker = speaker
                cur_text = text
            elif speaker == cur_speaker:
                cur_text += " " + text
            else:
                merged.append((cur_speaker, cur_text.strip()))
                cur_speaker = speaker
                cur_text = text
        if cur_speaker is not None:
            merged.append((cur_speaker, cur_text.strip()))
        return merged

    def _on_press(self, key):
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            now = time.monotonic()
            if now - self.last_ctrl_time <= self.config["double_press_interval"]:
                self.last_ctrl_time = 0.0
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            else:
                self.last_ctrl_time = now

    def _on_release(self, key):
        pass

    def run(self):
        self.load_model()
        with keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        ) as listener:
            try:
                self.stop_event.wait()
            except KeyboardInterrupt:
                print("\nВыход...")
            finally:
                listener.stop()
                self.executor.shutdown(wait=False)