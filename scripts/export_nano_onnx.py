#!/usr/bin/env python3
"""Export Qlerqly/griot-nano-1 (Conformer CTC) to ONNX for Sherpa-ONNX.

Model Card: https://huggingface.co/Qlerqly/griot-nano-1
Architecture: 153M Conformer CTC acoustic model for Ghanaian languages (Akan, Ewe, Ga, Dagbani, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


class GriotNanoOnnxWrapper(torch.nn.Module):
    """Clean ONNX export wrapper for Griot-Nano-1."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass for Sherpa-ONNX / WeNet CTC compatibility.

        Args:
            speech: [batch_size, time_frames, n_mels=80] log-mel filterbank
            speech_lengths: [batch_size] sequence lengths in frames

        Returns:
            log_probs: [batch_size, output_time_frames, vocab_size=50]
        """
        output = self.model(speech, speech_lengths)
        return output.log_probs


def export_griot_nano(
    model_id: str = "Qlerqly/griot-nano-1",
    output_dir: str = "models/griot-nano-1",
    opset: int = 17,
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model snapshot for '{model_id}'...")
    repo_dir = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["config.json", "vocab.json", "model.safetensors", "src/**"],
        )
    )

    # Import bespoke model architecture from repo
    sys.path.insert(0, str(repo_dir / "src"))
    from conformer_ctc.model import ConformerCTC, ConformerCTCConfig

    config_path = repo_dir / "config.json"
    vocab_path = repo_dir / "vocab.json"
    weights_path = repo_dir / "model.safetensors"

    config_dict = json.loads(config_path.read_text())
    vocab = json.loads(vocab_path.read_text())

    config = ConformerCTCConfig(**config_dict)
    model = ConformerCTC(config)
    model.load_state_dict(load_file(str(weights_path)))
    model.eval()

    # Generate tokens.txt for sherpa-onnx
    tokens_file = out_path / "tokens.txt"
    with open(tokens_file, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token} {idx}\n")
    print(f"Exported tokens mapping to {tokens_file} ({len(vocab)} tokens)")

    wrapper = GriotNanoOnnxWrapper(model)
    onnx_file = out_path / "model.onnx"

    # Dummy inputs: Batch=1, 100 frames, 80 mels
    dummy_speech = torch.randn(1, 100, 80, dtype=torch.float32)
    dummy_lens = torch.tensor([100], dtype=torch.long)

    print(f"Exporting ONNX model to {onnx_file} (opset {opset})...")
    torch.onnx.export(
        wrapper,
        (dummy_speech, dummy_lens),
        str(onnx_file),
        input_names=["speech", "speech_lengths"],
        output_names=["log_probs"],
        dynamic_axes={
            "speech": {0: "batch_size", 1: "time"},
            "speech_lengths": {0: "batch_size"},
            "log_probs": {0: "batch_size", 1: "output_time"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Model exported successfully! ({onnx_file.stat().st_size / (1024*1024):.1f} MB)")

    # Verification with ONNX Runtime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "speech": dummy_speech.numpy(),
            "speech_lengths": dummy_lens.numpy(),
        }
        ort_out = sess.run(None, ort_inputs)[0]

        with torch.no_grad():
            torch_out = wrapper(dummy_speech, dummy_lens).numpy()

        max_diff = np.max(np.abs(ort_out - torch_out))
        print(f"ONNX Runtime Parity Verification: Max absolute difference = {max_diff:.6f}")
        assert max_diff < 1e-3, f"Discrepancy too large: {max_diff}"
        print("Verification PASSED!")
    except ImportError:
        print("Note: onnxruntime not installed, skipping numerical verification.")

    return onnx_file


def main():
    parser = argparse.ArgumentParser(description="Export Griot-Nano-1 to ONNX for Sherpa-ONNX")
    parser.add_argument("--model-id", type=str, default="Qlerqly/griot-nano-1")
    parser.add_argument("--output-dir", type=str, default="models/griot-nano-1")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_griot_nano(args.model_id, args.output_dir, args.opset)


if __name__ == "__main__":
    main()
