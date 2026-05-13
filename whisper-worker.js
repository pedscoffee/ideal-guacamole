// whisper-worker.js — runs entirely in a Web Worker, zero main-thread blocking
// Uses @huggingface/transformers v3 via CDN. Audio never leaves the device.

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowLocalModels = false;
env.useBrowserCache = true;

// whisper-base.en: ~145MB, English-only, fast on M3
// No explicit dtype — Transformers.js auto-selects q8 for WASM, which is
// the quantization guaranteed to ship in this model's HuggingFace repo.
// Specifying dtype: { decoder_model_merged: "q4" } caused 404s because
// that shard filename doesn't exist for onnx-community/whisper-base.en.
const MODEL_ID = "onnx-community/whisper-base.en";

let transcriber = null;

async function loadModel(onProgress) {
  transcriber = await pipeline(
    "automatic-speech-recognition",
    MODEL_ID,
    {
      device: "wasm",
      // dtype intentionally omitted — let Transformers.js use the WASM
      // default (q8), which is the only quantization confirmed present
      // in the onnx-community/whisper-base.en repo.
      progress_callback: onProgress,
    }
  );
}

self.onmessage = async (e) => {
  const { type, audio, sampleRate } = e.data;

  if (type === "load") {
    try {
      await loadModel((p) => self.postMessage({ type: "progress", data: p }));
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    }
    return;
  }

  if (type === "transcribe") {
    if (!transcriber) {
      self.postMessage({ type: "error", message: "Model not loaded yet." });
      return;
    }
    try {
      // Explicit slice so the pipeline owns a clean buffer —
      // prevents "subarray is not a function" if the array arrived
      // via structured clone in an unusual state.
      const safeAudio = audio instanceof Float32Array
        ? audio.slice(0)
        : new Float32Array(audio);

      const result = await transcriber(
        { array: safeAudio, sampling_rate: sampleRate },
        {
          chunk_length_s:    30,
          stride_length_s:   5,
          language:          "en",
          task:              "transcribe",
          return_timestamps: false,
        }
      );

      self.postMessage({ type: "transcript", text: result.text.trim() });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    }
    return;
  }
};
