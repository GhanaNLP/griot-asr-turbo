# Deploying Griot ASR on Android with Sherpa-ONNX

This guide explains how to deploy **Griot-Nano-1** and **Griot-Edge** on Android devices for offline, on-device speech recognition in Akan (Asante/Akuapem Twi, Fante), Ewe, Ga, and Dagbani.

---

## 1. Add Sherpa-ONNX Dependency

In your Android project's `app/build.gradle.kts` (or `app/build.gradle`):

```kotlin
dependencies {
    // Sherpa-ONNX Android library
    implementation("com.k2fsa.sherpa.onnx:sherpa-onnx:1.10.30")
}
```

---

## 2. Copy Models to Assets

1. Quantize the model using `scripts/quantize_onnx.py` to minimize size:
   ```bash
   python scripts/quantize_onnx.py models/griot-nano-1/model.onnx -o model.int8.onnx
   ```
2. Place the following files in `app/src/main/assets/griot/`:
   - `model.int8.onnx` (or `model.onnx`)
   - `tokens.txt`

---

## 3. Kotlin Implementation

```kotlin
package com.example.ghananlp.asr

import android.content.Context
import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.*

class GriotAsrEngine(private val context: Context) {
    private var recognizer: OfflineRecognizer? = null

    init {
        initModel()
    }

    private fun initModel() {
        val assetManager = context.assets

        // 1. Configure Feature Extractor (80-channel log-mel filterbank)
        val featConfig = FeatureExtractorConfig(
            samplingRate = 16000,
            featureDim = 80
        )

        // 2. Configure Model
        val modelConfig = OfflineModelConfig(
            wenetCtc = OfflineWenetCtcModelConfig(
                model = "griot/model.int8.onnx"
            ),
            tokens = "griot/tokens.txt",
            numThreads = 2,
            provider = "cpu", // or "nnapi" on Android
            debug = false
        )

        // 3. Configure Offline Recognizer
        val config = OfflineRecognizerConfig(
            featConfig = featConfig,
            modelConfig = modelConfig,
            decodingMethod = "greedy_search"
        )

        recognizer = OfflineRecognizer(assetManager, config)
    }

    fun transcribe(samples: FloatArray, sampleRate: Int = 16000): String {
        val stream = recognizer?.createStream() ?: return ""
        stream.acceptWaveform(sampleRate, samples)
        recognizer?.decode(stream)
        val result = recognizer?.getResult(stream)?.text ?: ""
        stream.release()
        return result
    }

    fun release() {
        recognizer?.release()
        recognizer = null
    }
}
```

---

## 4. Hardware Acceleration (NNAPI)
On modern Android devices, you can optionally enable hardware acceleration:
```kotlin
val modelConfig = OfflineModelConfig(
    wenetCtc = OfflineWenetCtcModelConfig(model = "griot/model.int8.onnx"),
    tokens = "griot/tokens.txt",
    numThreads = 2,
    provider = "nnapi" // Android Neural Networks API
)
```

---

## 5. Performance Metrics on Mobile
- **Model Size:** ~60 MB (`griot-edge.int8.onnx`), ~150 MB (`griot-nano-1.int8.onnx`)
- **Latency:** ~0.15s per second of speech on mid-range ARM64 devices (6x real-time)
- **RAM Footprint:** ~80–120 MB
