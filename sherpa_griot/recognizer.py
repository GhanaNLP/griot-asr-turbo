"""Sherpa-ONNX wrapper for Griot Ghanaian Speech Recognition models."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Union

import numpy as np
import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None


class GriotRecognizer:
    """Offline ASR engine for Griot models powered by sherpa-onnx."""

    def __init__(
        self,
        model_path: Union[str, Path],
        tokens_path: Union[str, Path],
        num_threads: int = 4,
        provider: str = "cpu",
        debug: bool = False,
    ):
        if sherpa_onnx is None:
            raise ImportError("sherpa-onnx is required. Install via `pip install sherpa-onnx`.")

        self.model_path = Path(model_path)
        self.tokens_path = Path(tokens_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {self.tokens_path}")

        # Configure 80-channel log-mel filterbank
        feat_config = sherpa_onnx.FeatureExtractorConfig(
            sampling_rate=16000,
            feature_dim=80,
            low_freq=20,
            high_freq=-400,
            dither=0.0,
            normalize_samples=True,
            snip_edges=False,
        )

        # Griot is a Conformer CTC model compatible with WeNet CTC format
        model_config = sherpa_onnx.OfflineModelConfig(
            wenet_ctc=sherpa_onnx.OfflineWenetCtcModelConfig(model=str(self.model_path)),
            tokens=str(self.tokens_path),
            num_threads=num_threads,
            provider=provider,
            debug=debug,
        )

        recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoding_method="greedy_search",
        )

        self.recognizer = sherpa_onnx.OfflineRecognizer(recognizer_config)

    def transcribe_array(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe in-memory float32 audio samples."""
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Resample to 16000 Hz if needed
        if sample_rate != 16000:
            import math
            from scipy.signal import resample_poly
            g = math.gcd(int(sample_rate), 16000)
            samples = resample_poly(samples, 16000 // g, int(sample_rate) // g).astype(np.float32)
            sample_rate = 16000

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples.astype(np.float32))
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_file(self, audio_path: Union[str, Path]) -> str:
        """Transcribe an audio file (WAV, MP3, FLAC, OGG)."""
        samples, sr = sf.read(str(audio_path), dtype="float32")
        return self.transcribe_array(samples, sample_rate=sr)

    def transcribe_batch(self, audio_files: List[Union[str, Path]]) -> List[str]:
        """Transcribe a list of audio files."""
        results = []
        for path in audio_files:
            results.append(self.transcribe_file(path))
        return results
