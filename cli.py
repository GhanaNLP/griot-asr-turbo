#!/usr/bin/env python3
"""Command-line interface for Griot speech recognition with Sherpa-ONNX."""

import argparse
import sys
import time
from pathlib import Path

from sherpa_griot import GriotRecognizer


def main():
    parser = argparse.ArgumentParser(description="Transcribe Ghanaian audio files with Sherpa-ONNX")
    parser.add_argument("audio", type=str, nargs="+", help="Path to one or more audio files")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to the .onnx model file (e.g. models/griot-nano-1/model.onnx)",
    )
    parser.add_argument(
        "--tokens",
        "-t",
        type=str,
        required=True,
        help="Path to tokens.txt (e.g. tokens/tokens_nano.txt or tokens/tokens_edge.txt)",
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads (default: 4)")
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda", "coreml"], help="Execution provider")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    t0 = time.time()
    recognizer = GriotRecognizer(
        model_path=args.model,
        tokens_path=args.tokens,
        num_threads=args.threads,
        provider=args.provider,
    )
    print(f"Loaded in {time.time() - t0:.2f}s\n")

    for audio_path in args.audio:
        p = Path(audio_path)
        if not p.exists():
            print(f"Error: {p} does not exist", file=sys.stderr)
            continue

        t_start = time.time()
        transcript = recognizer.transcribe_file(p)
        elapsed = time.time() - t_start

        print(f"[{p.name}] ({elapsed:.2f}s):")
        print(f"  {transcript}\n")


if __name__ == "__main__":
    main()
