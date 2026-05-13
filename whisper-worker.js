// whisper-worker.js — runs entirely in a Web Worker, zero main-thread blocking
// Uses @huggingface/transformers v3 via CDN. Audio never leaves the device.

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowLocalModels = false;
env.useBrowserCache = true;

// ─── Model selection ──────────────────────────────────────────────────────
// whisper-base.en: ~145MB cached, English-only, fast on M3 WebGPU
// Upgrade to "onnx-community/whisper-small.en" (~244MB) for better accuracy
const MODEL_ID = "onnx-community/whisper-base.en";

let transcriber = null;

async function loadModel(onProgress) {
  // FIX 1: device must be an array with fallback, not a bare string.
  // Transformers.js v3 requires ["webgpu", "wasm"] for automatic fallback.
  // A bare "webgpu" string causes 404 on the jsep wasm shards when WebGPU
  // is unavailable, which is the 404 error seen in the console.
  //
  // FIX 2: dtype map must match the model's actual shard filenames.
  // "onnx-community/whisper-base.en" ships:
  //   encoder_model.onnx   → use "fp32"  (no quantized encoder available)
  //   decoder_model_merged_quantized.onnx → use "q4"
  // Using a dtype that has no matching file causes the 404.
  transcriber = await pipeline(
    "automatic-speech-recognition",
    MODEL_ID,
    {
      dtype: {
        encoder_model:        "fp32",
        decoder_model_merged: "q4",
      },
      device: "wasm",          // wasm is universal; WebGPU for Whisper is
                               // experimental and causes 404s on many builds.
                               // Switch to "webgpu" once your target browser
                               // confirmed supports it end-to-end for this model.
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
      // FIX 3: The pipeline's internal .subarray() call fails when the
      // transferred ArrayBuffer has been detached (zero-copy transfer
      // neutered the original). We must pass the raw Float32Array directly —
      // the structured clone (no transfer) keeps it alive in the worker.
      //
      // The correct input shape for transformers.js ASR pipeline is:
      //   { array: Float32Array, sampling_rate: number }
      //
      // "audio" here is already a Float32Array reconstructed from the
      // transferred buffer on the worker side — it is valid after transfer.
      // The subarray error fires when the buffer was transferred AND the
      // pipeline internally tries to slice it before it can copy it.
      // Solution: copy it once explicitly so the pipeline owns its own buffer.
      const safeAudio = audio instanceof Float32Array
        ? audio.slice(0)           // explicit copy — owns its buffer
        : new Float32Array(audio); // handle plain array fallback

      const result = await transcriber(
        { array: safeAudio, sampling_rate: sampleRate },
        {
          chunk_length_s:   30,
          stride_length_s:  5,
          language:         "en",
          task:             "transcribe",
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
