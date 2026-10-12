#!/usr/bin/env python3
"""Export Qlerqly/griot-edge (Parallel-Conformer CTC) to ONNX for Sherpa-ONNX.

Model Card: https://huggingface.co/Qlerqly/griot-edge
Architecture: 60.9M Streaming Parallel-Conformer CTC for Ghanaian Languages
Supported Languages: Akan (Asante/Akuapem Twi, Fante), Ewe, Ga, Dagbani, Dagaare, Ghanaian English
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download


class GriotEdgeOnnxWrapper(torch.nn.Module):
    """Sherpa-ONNX compatible wrapper for Griot-Edge."""

    def __init__(self, model: torch.nn.Module, lookahead_frames: int = 13):
        super().__init__()
        self.model = model
        self.lookahead_frames = lookahead_frames

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass for Sherpa-ONNX / WeNet CTC compatibility.

        Args:
            speech: [batch_size, time_frames, n_mels=80] log-mel filterbank
            speech_lengths: [batch_size] sequence lengths in frames

        Returns:
            log_probs: [batch_size, output_time_frames, vocab_size=151]
        """
        encoder_out, _ = self.model.encode(
            speech, speech_lengths, lookahead_frames=self.lookahead_frames
        )
        logits = self.model.ctc_head(encoder_out)
        return F.log_softmax(logits, dim=-1)


def export_griot_edge(
    model_id: str = "Qlerqly/griot-edge",
    output_dir: str = "models/griot-edge",
    opset: int = 17,
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model snapshot for '{model_id}'...")
    repo_dir = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["config.json", "vocab.json", "model.safetensors", "inference.py", "runtime.py"],
        )
    )

    # Import runtime from repo
    sys.path.insert(0, str(repo_dir))
    spec = importlib.util.spec_from_file_location("griot_inference", repo_dir / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("Loading PyTorch model...")
    model, vocab, _ = mod.load_model(repo_dir, device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    # Generate tokens.txt for sherpa-onnx
    tokens_file = out_path / "tokens.txt"
    with open(tokens_file, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token} {idx}\n")
    print(f"Exported tokens mapping to {tokens_file} ({len(vocab)} tokens)")

    rc_path = repo_dir / "release_config.json"
    lookahead = 13
    if rc_path.exists():
        rc = json.loads(rc_path.read_text())
        lookahead = rc.get("default_lookahead_frames", 13)

    wrapper = GriotEdgeOnnxWrapper(model, lookahead_frames=lookahead)
    onnx_file = out_path / "model.onnx"

    # Dummy input: 1 audio, 100 frames, 80 mel bins
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
    parser = argparse.ArgumentParser(description="Export Griot-Edge to ONNX for Sherpa-ONNX")
    parser.add_argument("--model-id", type=str, default="Qlerqly/griot-edge")
    parser.add_argument("--output-dir", type=str, default="models/griot-edge")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_griot_edge(args.model_id, args.output_dir, args.opset)


if __name__ == "__main__":
    main()
