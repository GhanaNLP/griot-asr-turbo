# Griot ASR Turbo 🇬🇭⚡

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Sherpa-ONNX](https://img.shields.io/badge/Powered%20by-sherpa--onnx-orange.svg)](https://github.com/k2-fsa/sherpa-onnx)
[![GhanaNLP](https://img.shields.io/badge/Organization-GhanaNLP-green.svg)](https://github.com/GhanaNLP)

Ultra-fast, offline, and lightweight Speech Recognition for **Ghanaian Languages** (Akan/Twi, Ewe, Ga, Dagbani, Dagaare, Ghanaian English) using **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)**.

Deploy state-of-the-art Conformer CTC speech models on **Android**, **iOS**, **Raspberry Pi**, **embedded devices**, and **servers** with zero internet connection required.

---

## 🌟 Benchmark Performance

According to the official [Nsanku Ghanaian ASR Benchmark](https://huggingface.co/spaces/ghananlpcommunity/nsanku-asr-benchmark) on Asante Twi:

| Model | Parameters | Asante Twi CER | Asante Twi WER | Recommended For |
|---|---|---|---|---|
| **`Qlerqly/griot-nano-1`** | 153M (0.2B) | **0.2212** (#1) | 0.4472 | Highest character accuracy, medical/loan words |
| **`Qlerqly/griot-edge`** | 60.9M (0.1B) | 0.2230 (#2) | **0.3738** (#1) | Lowest word error rate, ultra-fast mobile/embedded |

Both models achieve industry-leading performance on African speech recognition, beating larger multi-billion parameter models while remaining compact enough to run smoothly on edge hardware.

---

## 🚀 Quickstart (Python)

### 1. Installation

```bash
git clone https://github.com/GhanaNLP/griot-asr-turbo.git
cd griot-asr-turbo
pip install -r requirements.txt
```

### 2. Export Models to ONNX

Export the pre-trained weights to ONNX format:

```bash
# Export Griot-Nano-1 (Conformer CTC)
python scripts/export_nano_onnx.py --output-dir models/griot-nano-1

# Export Griot-Edge (Parallel Conformer CTC)
python scripts/export_edge_onnx.py --output-dir models/griot-edge
```

*(Optional)* Quantize to INT8 for a 4x reduction in model size (ideal for mobile deployment):
```bash
python scripts/quantize_onnx.py models/griot-nano-1/model.onnx -o models/griot-nano-1/model.int8.onnx
```

### 3. Transcribe Audio

#### Zero-Config CLI (Defaults to INT8 Griot-Nano-1):
```bash
# Single file
python cli.py audio_sample.wav

# Batch of multiple files (Parallel multi-stream decoding)
python cli.py audio1.wav audio2.wav audio3.wav --batch-size 16
```

#### Transcribe Hugging Face Datasets in Batch Mode:
```bash
python cli.py \
  --dataset ghananlpcommunity/twi-health-asr-gemini-500hrs-ipa \
  --split train \
  --limit 500 \
  --batch-size 32 \
  --output-parquet enriched_transcripts.parquet
```

#### Using Python Library:
```python
from sherpa_griot import GriotRecognizer

# 1. Initialize recognizer (defaults to pre-quantized INT8 Griot-Nano-1)
recognizer = GriotRecognizer()

# 2. Single-file transcription
text = recognizer.transcribe_file("twi_sample.wav")
print(f"Transcription: {text}")

# 3. Batch transcription for Hugging Face Datasets
from datasets import load_dataset
ds = load_dataset("ghananlpcommunity/twi-health-asr-gemini-500hrs-ipa", split="train")

# Automatically decodes in parallel batches and attaches 'griot_transcription' column
ds_enriched = recognizer.transcribe_dataset(ds, batch_size=32)
```

---

## 📱 Mobile & Edge Deployment

This repository formats the models specifically for direct use with `sherpa-onnx` mobile and embedded runtimes:

- 📱 **[Android Deployment Guide](docs/ANDROID.md)**: Add offline speech recognition to Android apps with Kotlin and Jetpack Compose.
- 🍏 **[iOS / Swift Deployment Guide](docs/IOS.md)**: Run offline on iPhones and iPads with CoreML hardware acceleration.
- 🍓 **[Embedded Linux & Raspberry Pi Guide](docs/EMBEDDED.md)**: Run standalone using pre-compiled C++ binaries with zero Python dependencies.

---

## 📁 Repository Structure

```
griot-sherpa-onnx/
├── docs/                      # Mobile and embedded deployment guides
│   ├── ANDROID.md             # Android Kotlin integration
│   ├── IOS.md                 # iOS Swift integration
│   └── EMBEDDED.md            # C/C++ and Raspberry Pi setup
├── scripts/
│   ├── export_nano_onnx.py    # PyTorch -> ONNX exporter for Griot-Nano-1
│   ├── export_edge_onnx.py    # PyTorch -> ONNX exporter for Griot-Edge
│   └── quantize_onnx.py       # INT8 dynamic quantization script
├── sherpa_griot/              # Lightweight Python library
│   ├── __init__.py
│   └── recognizer.py          # GriotRecognizer wrapper
├── tokens/
│   ├── tokens_nano.txt        # Token dictionary for Nano-1 (50 tokens)
│   └── tokens_edge.txt        # Token dictionary for Edge (151 tokens)
├── cli.py                     # Command-line transcription tool
├── requirements.txt
└── pyproject.toml
```

---

## 🌍 Supported Languages

The models and tokenizers support:
- **Akan**: Asante Twi, Akuapem Twi, Fante
- **Ewe**
- **Ga**
- **Dagbani**
- **Dagaare**
- **Ghanaian English**

---

## 📄 License & Attribution

- **Code:** Licensed under [Apache 2.0](LICENSE).
- **Models:** Model weights are developed and provided by [Qlerqly](https://huggingface.co/Qlerqly).
- **Engine:** Built on top of the open-source [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) framework by next-gen Kaldi.
