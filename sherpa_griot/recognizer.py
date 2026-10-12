"""Sherpa-ONNX wrapper for Griot Ghanaian Speech Recognition models with Batch & HF support."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = DEFAULT_ROOT / "models" / "griot-nano-1" / "model.int8.onnx"
DEFAULT_TOKENS = DEFAULT_ROOT / "tokens" / "tokens_nano.txt"


def _resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Ensure float32 mono audio at 16000 Hz."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != 16000:
        g = math.gcd(int(sample_rate), 16000)
        samples = resample_poly(samples, 16000 // g, int(sample_rate) // g)
    return np.ascontiguousarray(samples, dtype=np.float32)


class GriotRecognizer:
    """Offline ASR engine for Griot models powered by sherpa-onnx.

    Supports single-file, batch multi-stream inference, and Hugging Face dataset processing.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tokens_path: Optional[Union[str, Path]] = None,
        num_threads: int = 4,
        provider: str = "cpu",
        debug: bool = False,
    ):
        if sherpa_onnx is None:
            raise ImportError("sherpa-onnx is required. Install via `pip install sherpa-onnx`.")

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self.tokens_path = Path(tokens_path) if tokens_path else DEFAULT_TOKENS

        if not self.model_path.exists():
            # Fallback to FP32 if INT8 not yet generated
            fp32_model = self.model_path.with_name("model.onnx")
            if fp32_model.exists():
                self.model_path = fp32_model
            else:
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
        samples_16k = _resample_to_16k(samples, sample_rate)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, samples_16k)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_file(self, audio_path: Union[str, Path]) -> str:
        """Transcribe a single audio file (WAV, MP3, FLAC, OGG)."""
        samples, sr = sf.read(str(audio_path), dtype="float32")
        return self.transcribe_array(samples, sample_rate=sr)

    def transcribe_batch_arrays(
        self, audio_list: List[np.ndarray], sample_rate: Union[int, List[int]] = 16000
    ) -> List[str]:
        """Transcribe a batch of audio arrays concurrently using sherpa-onnx multi-stream decoding."""
        if not audio_list:
            return []

        if isinstance(sample_rate, int):
            rates = [sample_rate] * len(audio_list)
        else:
            rates = sample_rate

        streams = []
        for arr, sr in zip(audio_list, rates):
            samples_16k = _resample_to_16k(arr, sr)
            s = self.recognizer.create_stream()
            s.accept_waveform(16000, samples_16k)
            streams.append(s)

        # Parallel multi-stream C++ decoding inside Sherpa-ONNX
        self.recognizer.decode_streams(streams)
        return [s.result.text.strip() for s in streams]

    def transcribe_batch_files(self, audio_files: List[Union[str, Path]], batch_size: int = 32) -> List[str]:
        """Transcribe multiple audio files in parallel batches."""
        results: List[str] = []
        for i in range(0, len(audio_files), batch_size):
            chunk = audio_files[i : i + batch_size]
            audio_arrays = []
            rates = []
            for path in chunk:
                arr, sr = sf.read(str(path), dtype="float32")
                audio_arrays.append(arr)
                rates.append(sr)
            transcripts = self.transcribe_batch_arrays(audio_arrays, sample_rate=rates)
            results.extend(transcripts)
        return results

    def transcribe_dataset(
        self,
        dataset: Any,
        audio_column: str = "audio",
        output_column: str = "griot_transcription",
        batch_size: int = 32,
        progress_callback: Optional[Any] = None,
    ) -> Any:
        """Batch-transcribe a Hugging Face Dataset or IterableDataset.

        Supports audio delivered as:
          - Dict with 'bytes' (e.g. datasets.Audio(decode=False))
          - Dict with 'array' (e.g. datasets.Audio(decode=True))
          - Dict with 'path' / string file path

        Returns:
            Updated dataset with the new `output_column` attached.
        """

        def extract_audio(sample_item: Dict[str, Any]) -> Tuple[np.ndarray, int]:
            val = sample_item.get(audio_column)
            if isinstance(val, dict):
                if val.get("bytes"):
                    with io.BytesIO(val["bytes"]) as f:
                        arr, sr = sf.read(f, dtype="float32")
                    return arr, sr
                if "array" in val:
                    return np.asarray(val["array"], dtype=np.float32), val.get("sampling_rate", 16000)
                if val.get("path"):
                    arr, sr = sf.read(val["path"], dtype="float32")
                    return arr, sr
            elif isinstance(val, str):
                arr, sr = sf.read(val, dtype="float32")
                return arr, sr
            raise ValueError(f"Unable to extract audio from sample column '{audio_column}': {val}")

        # Check if dataset is iterable or map-style
        is_iterable = hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__")

        if is_iterable:
            def generator() -> Iterator[Dict[str, Any]]:
                buffer: List[Dict[str, Any]] = []
                for item in dataset:
                    buffer.append(item)
                    if len(buffer) >= batch_size:
                        arrays, srs = [], []
                        for b in buffer:
                            a, s = extract_audio(b)
                            arrays.append(a)
                            srs.append(s)
                        texts = self.transcribe_batch_arrays(arrays, sample_rate=srs)
                        for b, t in zip(buffer, texts):
                            b[output_column] = t
                            yield b
                        buffer = []
                if buffer:
                    arrays, srs = [], []
                    for b in buffer:
                        a, s = extract_audio(b)
                        arrays.append(a)
                        srs.append(s)
                    texts = self.transcribe_batch_arrays(arrays, sample_rate=srs)
                    for b, t in zip(buffer, texts):
                        b[output_column] = t
                        yield b

            return generator()

        # Map-style dataset
        def batch_process(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            items_count = len(batch[audio_column])
            audio_arrays, srs = [], []
            for j in range(items_count):
                single_audio = {audio_column: batch[audio_column][j]}
                a, s = extract_audio(single_audio)
                audio_arrays.append(a)
                srs.append(s)
            texts = self.transcribe_batch_arrays(audio_arrays, sample_rate=srs)
            batch[output_column] = texts
            if progress_callback:
                progress_callback(items_count)
            return batch

        return dataset.map(batch_process, batched=True, batch_size=batch_size)
