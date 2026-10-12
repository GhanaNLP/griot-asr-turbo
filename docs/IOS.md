# Deploying Griot ASR on iOS with Sherpa-ONNX

This guide details how to integrate **Griot-Nano-1** and **Griot-Edge** into an iOS app using Swift and `sherpa-onnx` for offline on-device transcription of Ghanaian languages.

---

## 1. Add Swift Package Dependency

In Xcode:
1. Go to **File** > **Add Packages...**
2. Enter the repository URL: `https://github.com/k2-fsa/sherpa-onnx`
3. Add the package to your app target.

---

## 2. Add Models to Xcode Project

1. Export and optionally quantize your model:
   ```bash
   python scripts/quantize_onnx.py models/griot-edge/model.onnx -o model.int8.onnx
   ```
2. Drag and drop `model.int8.onnx` and `tokens.txt` into your Xcode project (ensure **Copy items if needed** and **Target Membership** are checked).

---

## 3. Swift Implementation

```swift
import Foundation
import sherpa_onnx

class GriotAsrService {
    private var recognizer: OfflineRecognizer?

    init() {
        guard let modelPath = Bundle.main.path(forResource: "model.int8", ofType: "onnx"),
              let tokensPath = Bundle.main.path(forResource: "tokens", ofType: "txt") else {
            print("Failed to find model files in main bundle.")
            return
        }

        var featConfig = sherpaOnnxFeatureExtractorConfig()
        featConfig.samplingRate = 16000
        featConfig.featureDim = 80

        var modelConfig = sherpaOnnxOfflineModelConfig()
        modelConfig.wenetCtc.model = modelPath
        modelConfig.tokens = tokensPath
        modelConfig.numThreads = 2
        modelConfig.provider = "coreml" // or "cpu"

        var config = sherpaOnnxOfflineRecognizerConfig()
        config.featConfig = featConfig
        config.modelConfig = modelConfig
        config.decodingMethod = "greedy_search"

        self.recognizer = OfflineRecognizer(config: &config)
    }

    func transcribe(samples: [Float], sampleRate: Int = 16000) -> String {
        guard let recognizer = self.recognizer else { return "" }
        let stream = recognizer.createStream()
        stream.acceptWaveform(samples: samples, sampleRate: Int32(sampleRate))
        recognizer.decode(stream: stream)
        let result = recognizer.getResult(stream: stream).text
        return result
    }
}
```

---

## 4. Hardware Acceleration (CoreML)
On Apple Silicon / iOS devices (iPhone, iPad), you can enable the Neural Engine via `CoreML`:
```swift
modelConfig.provider = "coreml"
```
This reduces battery consumption and provides near-instantaneous transcription of Ghanaian speech offline.
