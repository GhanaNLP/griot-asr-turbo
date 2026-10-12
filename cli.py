#!/usr/bin/env python3
"""Command-line interface for Griot speech recognition with Sherpa-ONNX.

Supports single files, batch audio files, and Hugging Face dataset processing.
"""

import argparse
import sys
import time
from pathlib import Path

from sherpa_griot import GriotRecognizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "griot-nano-1"
DEFAULT_MODEL = DEFAULT_MODEL_DIR / "model.int8.onnx"
DEFAULT_TOKENS = Path(__file__).resolve().parent / "tokens" / "tokens_nano.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Ghanaian speech (Twi, Ewe, Ga, Dagbani) with Griot ASR Turbo"
    )
    parser.add_argument(
        "audio",
        type=str,
        nargs="*",
        default=[],
        help="Path to one or more audio files (.wav, .mp3, .flac)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g. ghananlpcommunity/twi-health-asr-gemini-500hrs-ipa)",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--limit", type=int, default=0, help="Max dataset samples to transcribe (0 = all)")
    parser.add_argument(
        "--output-parquet",
        "-o",
        type=str,
        default=None,
        help="Output path to save transcribed dataset as Parquet",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=str(DEFAULT_MODEL),
        help=f"Path to .onnx model file (default: {DEFAULT_MODEL.name} [INT8 Nano-1])",
    )
    parser.add_argument(
        "--tokens",
        "-t",
        type=str,
        default=str(DEFAULT_TOKENS),
        help=f"Path to tokens.txt (default: {DEFAULT_TOKENS.name})",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for parallel decoding (default: 32)")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads (default: 4)")
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda", "coreml"], help="Execution provider")
    args = parser.parse_args()

    if not args.audio and not args.dataset:
        parser.print_help()
        sys.exit(1)

    model_p = Path(args.model)
    tokens_p = Path(args.tokens)

    if not model_p.exists():
        fp32_model = model_p.with_name("model.onnx")
        if fp32_model.exists():
            model_p = fp32_model
        else:
            print(f"Error: Model not found at '{model_p}'.", file=sys.stderr)
            print("Please export the model first via: python scripts/export_nano_onnx.py", file=sys.stderr)
            sys.exit(1)

    print(f"Model:    {model_p.name} (INT8 Griot-Nano-1)")
    print(f"Tokens:   {tokens_p.name}")
    print(f"Threads:  {args.threads} | Batch Size: {args.batch_size}")
    print("-" * 60)

    t0 = time.time()
    recognizer = GriotRecognizer(
        model_path=model_p,
        tokens_path=tokens_p,
        num_threads=args.threads,
        provider=args.provider,
    )
    print(f"Engine initialized in {time.time() - t0:.2f}s\n")

    # 1. Hugging Face Dataset Mode
    if args.dataset:
        try:
            import datasets
        except ImportError:
            print("Error: 'datasets' package is required for HF dataset mode. Run `pip install datasets`.", file=sys.stderr)
            sys.exit(1)

        print(f"Loading Hugging Face dataset: '{args.dataset}' (split: {args.split})...")
        ds = datasets.load_dataset(args.dataset, split=args.split)
        if args.limit > 0:
            ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Dataset loaded: {len(ds)} items. Beginning parallel batch transcription...")

        t_start = time.time()
        processed_count = [0]

        def on_progress(n):
            processed_count[0] += n
            pct = (processed_count[0] / len(ds)) * 100.0
            print(f"Progress: {processed_count[0]}/{len(ds)} ({pct:.1f}%)", end="\r", flush=True)

        ds_enriched = recognizer.transcribe_dataset(
            ds,
            batch_size=args.batch_size,
            progress_callback=on_progress,
        )
        total_time = time.time() - t_start
        print(f"\nTranscribed {len(ds)} samples in {total_time:.2f}s ({len(ds)/total_time:.1f} samples/s)!")

        if args.output_parquet:
            out_p = Path(args.output_parquet)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            ds_enriched.to_parquet(str(out_p))
            print(f"Saved enriched dataset to {out_p}")
        return

    # 2. Local Files Mode (Batch or Single)
    if len(args.audio) == 1:
        p = Path(args.audio[0])
        t_start = time.time()
        transcript = recognizer.transcribe_file(p)
        elapsed = time.time() - t_start
        print(f"[{p.name}] (processed in {elapsed:.2f}s):")
        print(f"  {transcript}\n")
    else:
        print(f"Transcribing {len(args.audio)} audio files in batches of {args.batch_size}...")
        t_start = time.time()
        results = recognizer.transcribe_batch_files(args.audio, batch_size=args.batch_size)
        total_time = time.time() - t_start
        print(f"Done {len(args.audio)} files in {total_time:.2f}s!\n")
        for p, txt in zip(args.audio, results):
            print(f"[{Path(p).name}]:\n  {txt}\n")


if __name__ == "__main__":
    main()
