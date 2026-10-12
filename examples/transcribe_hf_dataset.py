#!/usr/bin/env python3
"""Example: Transcribing a Hugging Face Dataset with Griot ASR Turbo in Batch Mode.

Demonstrates parallel multi-stream decoding using sherpa-onnx on Hugging Face datasets.
"""

import time
from datasets import load_dataset
from sherpa_griot import GriotRecognizer


def main():
    # 1. Initialize Recognizer (Defaults to INT8 Griot-Nano-1)
    print("Loading Griot ASR Turbo...")
    recognizer = GriotRecognizer(num_threads=4)

    # 2. Load Audio Dataset from Hugging Face
    dataset_name = "ghananlpcommunity/twi-health-asr-gemini-500hrs-ipa"
    print(f"Streaming dataset '{dataset_name}'...")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # Take first 20 samples as an example batch
    samples = []
    for i, s in enumerate(ds):
        samples.append(s)
        if i >= 19:
            break

    print(f"Collected {len(samples)} audio samples.")

    # 3. Batch Transcribe with Parallel Multi-Stream Decoding
    t0 = time.time()
    enriched_samples = list(
        recognizer.transcribe_dataset(
            samples,
            audio_column="audio",
            output_column="griot_transcription",
            batch_size=8,
        )
    )
    elapsed = time.time() - t0

    print(f"\nFinished batch transcription in {elapsed:.2f}s ({len(samples)/elapsed:.1f} samples/sec)!\n")

    # 4. Display Results
    for item in enriched_samples[:3]:
        print(f"ID: {item.get('id')}")
        print(f"  Original:  {item.get('transcription', '')[:90]}...")
        print(f"  Griot-ASR: {item.get('griot_transcription', '')[:90]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
