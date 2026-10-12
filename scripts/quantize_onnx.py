#!/usr/bin/env python3
"""INT8 Dynamic Quantization for Sherpa-ONNX Models.

Reduces model footprint by ~4x (ideal for Android, iOS, Raspberry Pi, and Edge deployment).
"""

import argparse
from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(input_model: str, output_model: str | None = None) -> Path:
    in_path = Path(input_model)
    if output_model is None:
        out_path = in_path.with_name(in_path.stem + ".int8.onnx")
    else:
        out_path = Path(output_model)

    print(f"Quantizing {in_path} to INT8 dynamic...")
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )

    size_orig_mb = in_path.stat().st_size / (1024 * 1024)
    size_quant_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Original size:   {size_orig_mb:.1f} MB")
    print(f"Quantized size:  {size_quant_mb:.1f} MB ({100.0 * size_quant_mb / size_orig_mb:.1f}% of original)")
    print(f"Saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX ASR model to INT8")
    parser.add_argument("input_model", type=str, help="Path to input .onnx file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .int8.onnx path")
    args = parser.parse_args()

    quantize_model(args.input_model, args.output)


if __name__ == "__main__":
    main()
