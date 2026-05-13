// whisper-worker.js — runs entirely in a Web Worker, zero main-thread blocking
// Uses @huggingface/transformers v3 via CDN. Audio never leaves the device.

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

// Only use browser cache — no Node.js filesystem
env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "onnx-community/whisper-base.en";
// whisper-base.en: ~145MB download, English-only, fast on M3, good accuracy
// Upgrade path: whisper-small.en (~244MB) for better accuracy if needed

let transcriber = null;

async function loadModel(progressCallback) {
  transcriber = await pipeline(
    "automatic-speech-recognition",
    MODEL_ID,
    {
      dtype: {
        encoder_model: "fp32",
        decoder_model_merged: "q4",   // quantized decoder — fast, small
      },
      device: "webgpu",               // falls back to wasm automatically
      progress_callback: progressCallback,
    }
  );
}

self.onmessage = async (e) => {
  const { type, audio, sampleRate } = e.data;

  if (type === "load") {
    try {
      await loadModel((progress) => {
        self.postMessage({ type: "progress", data: progress });
      });
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
    return;
  }

  if (type === "transcribe") {
    if (!transcriber) {
      self.postMessage({ type: "error", message: "Model not loaded yet." });
      return;
    }
    try {
      // transformers.js expects Float32Array at 16kHz
      const result = await transcriber(
        { array: audio, sampling_rate: sampleRate },
        {
          chunk_length_s: 30,
          stride_length_s: 5,
          language: "en",
          task: "transcribe",
          return_timestamps: false,
        }
      );
      self.postMessage({ type: "transcript", text: result.text.trim() });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
    return;
  }
};
