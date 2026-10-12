# Embedded & C/C++ Deployment Guide

Griot models exported with this repository can run standalone using pre-compiled `sherpa-onnx` C++ binaries on **Raspberry Pi**, **NVIDIA Jetson**, **RISC-V**, and embedded Linux systems with zero Python dependencies.

---

## 1. Download Pre-compiled Sherpa-ONNX Binary

Download the appropriate release for your architecture from [Sherpa-ONNX Releases](https://github.com/k2-fsa/sherpa-onnx/releases):

```bash
# Example for Raspberry Pi 4 / 5 (ARM64 / aarch64 Linux):
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.30/sherpa-onnx-v1.10.30-linux-aarch64.tar.bz2
tar xvf sherpa-onnx-v1.10.30-linux-aarch64.tar.bz2
export PATH=$PATH:$PWD/sherpa-onnx-v1.10.30-linux-aarch64/bin
```

---

## 2. CLI Offline Speech Recognition

Run offline transcription directly in the terminal:

```bash
sherpa-onnx-offline \
  --tokens=tokens/tokens_nano.txt \
  --wenet-ctc=models/griot-nano-1/model.onnx \
  --num-threads=4 \
  input_audio.wav
```

With INT8 quantized model on Raspberry Pi:
```bash
sherpa-onnx-offline \
  --tokens=tokens/tokens_nano.txt \
  --wenet-ctc=models/griot-nano-1/model.int8.onnx \
  --num-threads=2 \
  input_audio.wav
```

---

## 3. C API Minimal Example

```c
#include "sherpa-onnx/c-api/c-api.h"
#include <stdio.h>

int main() {
    SherpaOnnxOfflineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;

    config.model_config.wenet_ctc.model = "model.int8.onnx";
    config.model_config.tokens = "tokens.txt";
    config.model_config.num_threads = 2;
    config.model_config.provider = "cpu";
    config.decoding_method = "greedy_search";

    const SherpaOnnxOfflineRecognizer *recognizer = SherpaOnnxCreateOfflineRecognizer(&config);
    const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(recognizer);

    // Feed audio samples (float array at 16kHz)
    // SherpaOnnxAcceptWaveformOffline(stream, 16000, samples, n_samples);

    SherpaOnnxDecodeOfflineStream(recognizer, stream);
    const SherpaOnnxOfflineRecognizerResult *result = SherpaOnnxGetOfflineStreamResult(stream);

    printf("Transcription: %s\n", result->text);

    SherpaOnnxDestroyOfflineRecognizerResult(result);
    SherpaOnnxDestroyOfflineStream(stream);
    SherpaOnnxDestroyOfflineRecognizer(recognizer);
    return 0;
}
```
