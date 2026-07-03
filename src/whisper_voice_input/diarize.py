"""Разделение спикеров на ECAPA-TDNN (speechbrain) + агломеративная кластеризация.

Модель `speechbrain/spkrec-ecapa-voxceleb` — открытая лицензия, скачивается
с HuggingFace без токена. Кэшируется в стандартной директории HF Hub.
"""

import sys
from pathlib import Path

import numpy as np


class Diarizer:
    def __init__(self, min_speakers=1, max_speakers=4, sample_rate=16000):
        self.min_speakers = max(1, int(min_speakers))
        self.max_speakers = max(self.min_speakers, int(max_speakers))
        self.sample_rate = sample_rate
        self.encoder = None
        self.torch = None

    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    MODEL_SAVEDIR = str(Path.home() / ".cache" / "whisper-voice-input" / "spkrec-ecapa-voxceleb")

    def load(self):
        try:
            import torch
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError as e:
            print(
                "Не удалось импортировать speechbrain/torch для диаризации.\n"
                "Установите extra: pip install -e .[diarize]\n"
                f"Детали: {e}",
                file=sys.stderr,
            )
            return False

        self.torch = torch
        try:
            self.encoder = SpeakerRecognition.from_hparams(
                source=self.MODEL_SOURCE,
                savedir=self.MODEL_SAVEDIR,
                run_opts={"device": "cpu"},
            )
        except Exception as e:
            print(
                f"Не удалось загрузить ECAPA-модель: {e}",
                file=sys.stderr,
            )
            return False
        return True

    def _segment_embedding(self, wav_chunk):
        if len(wav_chunk) < int(self.sample_rate * 0.3):
            return None
        wav = self.torch.from_numpy(wav_chunk).float().unsqueeze(0)
        with self.torch.no_grad():
            emb = self.encoder.encode_batch(wav)
        return emb.squeeze().cpu().numpy()

    def diarize(self, audio, segments):
        """Вернуть [(speaker_id, text), ...] для переданных whisper-сегментов.

        segments — список faster_whisper.transcribe.Segment с атрибутами
        start, end, text (время в секундах, текст str).
        """
        if not segments:
            return []

        sr = self.sample_rate
        embeddings = []
        valid_idx = []
        for i, seg in enumerate(segments):
            s = max(0, int(seg.start * sr))
            e = min(len(audio), int(seg.end * sr))
            if e <= s:
                embeddings.append(None)
                continue
            emb = self._segment_embedding(audio[s:e])
            embeddings.append(emb)
            if emb is not None:
                valid_idx.append(i)

        if not valid_idx:
            return [(0, getattr(seg, "text", "").strip()) for seg in segments]

        from sklearn.cluster import AgglomerativeClustering

        X = np.stack([embeddings[i] for i in valid_idx])

        if X.shape[0] == 1:
            labels_valid = np.array([0])
        else:
            n_clusters = self._choose_n_clusters(X)
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels_valid = clustering.fit_predict(X)

        labels = [0] * len(segments)
        for j, i in enumerate(valid_idx):
            labels[i] = int(labels_valid[j])

        for i, emb in enumerate(embeddings):
            if emb is None and labels[i] == 0:
                nearest = self._nearest_label(i, valid_idx, labels)
                if nearest is not None:
                    labels[i] = nearest

        result = []
        for i, seg in enumerate(segments):
            text = getattr(seg, "text", "").strip()
            if text:
                result.append((labels[i], text))
        return result

    def _choose_n_clusters(self, X):
        from sklearn.cluster import AgglomerativeClustering

        if self.min_speakers == self.max_speakers:
            return self.min_speakers

        n_samples = X.shape[0]
        upper = min(self.max_speakers, n_samples)
        lower = min(self.min_speakers, upper)
        if lower >= upper:
            return lower

        last_labels = None
        for k in range(upper, lower - 1, -1):
            clustering = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            )
            labels = clustering.fit_predict(X)
            if k == upper:
                last_labels = labels
            if self._cluster_separation_ok(X, labels):
                return k
        if last_labels is not None:
            unique = len(set(last_labels.tolist()))
            return max(lower, min(unique, upper))
        return lower

    @staticmethod
    def _cluster_separation_ok(X, labels):
        unique = set(labels.tolist())
        if len(unique) < 2:
            return True
        centroids = []
        for c in sorted(unique):
            mask = labels == c
            if mask.sum() == 0:
                continue
            centroids.append(X[mask].mean(axis=0))
        if len(centroids) < 2:
            return True
        centroids = np.stack(centroids)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = centroids / norms
        sim = unit @ unit.T
        np.fill_diagonal(sim, 0.0)
        max_sim = float(sim.max())
        return max_sim < 0.5

    @staticmethod
    def _nearest_label(idx, valid_idx, labels):
        for j in range(len(valid_idx)):
            if valid_idx[j] > idx:
                return labels[valid_idx[j]]
        for j in range(len(valid_idx) - 1, -1, -1):
            if valid_idx[j] < idx:
                return labels[valid_idx[j]]
        return None

    def num_speakers(self, tagged):
        return len({s for s, _ in tagged})