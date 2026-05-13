// Present — app.js
// WebLLM (Qwen3) + local Whisper (Transformers.js) for clinical A&P note generation
// Audio never leaves the browser.

import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// ─── State ────────────────────────────────────────────────────────────────
let engine = null;
let isModelReady = false;
let isRecording = false;
let transcript = "";
let timerInterval = null;
let timerSeconds = 0;
let currentTab = "mic";

// Whisper state
let whisperPipeline = null;
let isWhisperReady = false;
let isTranscribing = false;

// MediaRecorder state
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;

// ─── Model Setup ──────────────────────────────────────────────────────────
const MODEL_ID = "Qwen3-4B-q4f16_1-MLC";

async function initModel() {
  setStatus("loading", "Initializing LLM…");
  showProgress(true, "Downloading model weights (first load ~2.3GB, cached after)…", 0);

  try {
    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (progress) => {
        const pct = Math.round((progress.progress || 0) * 100);
        showProgress(true, progress.text || "Loading…", pct);
      }
    });
    isModelReady = true;
    setStatus("ready", "Model ready");
    showProgress(false);
    updateProcessBtn();
  } catch (err) {
    console.error("Model init failed:", err);
    setStatus("error", "Model failed to load");
    showProgress(false);
    showError("Failed to load the AI model. Please check your browser supports WebGPU (Chrome 113+ recommended) and reload.");
  }
}

// ─── Whisper Setup ────────────────────────────────────────────────────────
// Loads whisper-small.en locally via Transformers.js (WASM/WebGPU).
// Model weights (~244 MB) are cached in the browser after first load.
async function initWhisper() {
  setWhisperStatus("loading", "Loading Whisper…");
  try {
    // Use whisper-small.en for better accuracy; swap to whisper-base.en for faster load
    whisperPipeline = await pipeline(
      "automatic-speech-recognition",
      "Xenova/whisper-small.en",
      {
        progress_callback: (progress) => {
          if (progress.status === "downloading") {
            const pct = progress.total ? Math.round((progress.loaded / progress.total) * 100) : 0;
            setWhisperStatus("loading", `Loading Whisper… ${pct}%`);
          }
        }
      }
    );
    isWhisperReady = true;
    setWhisperStatus("ready", "Whisper ready (local)");
    const btn = document.getElementById("micBtn");
    if (btn) btn.disabled = false;
    const hint = document.getElementById("micHint");
    if (hint) hint.textContent = "Click to begin recording";
  } catch (err) {
    console.error("Whisper init failed:", err);
    setWhisperStatus("error", "Whisper failed to load — try reloading");
    const hint = document.getElementById("micHint");
    if (hint) hint.textContent = "Whisper failed to load. Please reload the page.";
  }
}

function setWhisperStatus(state, text) {
  const dot = document.getElementById("whisperDot");
  const label = document.getElementById("whisperText");
  if (!dot || !label) return;
  dot.className = "status-dot " + state;
  label.textContent = text;
}

// ─── UI helpers ───────────────────────────────────────────────────────────
function setStatus(state, text) {
  const dot = document.getElementById("statusDot");
  const label = document.getElementById("statusText");
  dot.className = "status-dot " + state;
  label.textContent = text;
}

function showProgress(visible, label = "", pct = 0) {
  const wrap = document.getElementById("progressWrap");
  const bar = document.getElementById("progressBar");
  const lbl = document.getElementById("progressLabel");
  if (visible) {
    wrap.style.display = "block";
    bar.style.width = pct + "%";
    lbl.textContent = label;
  } else {
    wrap.style.display = "none";
  }
}

function showError(msg) {
  const area = document.getElementById("outputArea");
  const empty = document.getElementById("outputEmpty");
  const content = document.getElementById("outputContent");
  const streaming = document.getElementById("outputStreaming");
  empty.style.display = "none";
  content.style.display = "none";
  streaming.style.display = "none";
  area.innerHTML = `<div class="error-msg">${msg}</div>`;
}

function updateProcessBtn() {
  const btn = document.getElementById("btnProcess");
  const hasContent = currentTab === "mic"
    ? transcript.trim().length > 20
    : document.getElementById("textInput").value.trim().length > 20;
  btn.disabled = !isModelReady || !hasContent;
}

// ─── Tab switching ────────────────────────────────────────────────────────
window.switchTab = function(tab) {
  currentTab = tab;
  document.getElementById("micTab").classList.toggle("hidden", tab !== "mic");
  document.getElementById("textTab").classList.toggle("hidden", tab !== "text");
  document.getElementById("tabMic").classList.toggle("active", tab === "mic");
  document.getElementById("tabText").classList.toggle("active", tab === "text");
  updateProcessBtn();
};

// ─── Audio Recording (MediaRecorder) ─────────────────────────────────────
// Records audio as WebM/OGG; on stop, decodes to PCM float32 for Whisper.

window.toggleRecording = async function() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
};

async function startRecording() {
  if (!isWhisperReady) {
    const hint = document.getElementById("micHint");
    if (hint) hint.textContent = "Whisper is still loading, please wait…";
    return;
  }

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    console.error("Microphone access denied:", err);
    const hint = document.getElementById("micHint");
    if (hint) hint.textContent = "Microphone access denied. Please allow microphone access and try again.";
    return;
  }

  // Pick best supported MIME type
  const mimeType = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg"
  ].find(t => MediaRecorder.isTypeSupported(t)) || "";

  audioChunks = [];
  mediaRecorder = new MediaRecorder(audioStream, mimeType ? { mimeType } : undefined);

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) audioChunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    // Stop all mic tracks to release the microphone indicator
    audioStream.getTracks().forEach(t => t.stop());

    // Build blob and transcribe
    const blob = new Blob(audioChunks, { type: mimeType || "audio/webm" });
    await transcribeAudio(blob);
  };

  mediaRecorder.start(250); // collect data every 250ms
  isRecording = true;

  document.getElementById("micVisualizer").classList.add("recording");
  document.getElementById("micBtn").style.cssText = "";
  document.getElementById("micHint").textContent = "Recording… click to stop";
  document.getElementById("recordingTimer").style.display = "flex";
  timerSeconds = 0;
  updateTimer();
  timerInterval = setInterval(updateTimer, 1000);
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  isRecording = false;
  clearInterval(timerInterval);
  document.getElementById("micVisualizer").classList.remove("recording");
  document.getElementById("micHint").textContent = "Transcribing with local Whisper…";
  document.getElementById("recordingTimer").style.display = "none";
  document.getElementById("micBtn").disabled = true;
  mediaRecorder.stop(); // triggers onstop → transcribeAudio
}

async function transcribeAudio(blob) {
  isTranscribing = true;
  const transcriptEl = document.getElementById("transcriptText");
  transcriptEl.innerHTML = '<span class="placeholder">Transcribing locally with Whisper…</span>';

  try {
    // Decode to AudioBuffer via Web Audio API
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    let audioBuffer;
    try {
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    } finally {
      audioCtx.close();
    }

    // Resample to 16 kHz mono float32 (Whisper's required format)
    const float32 = resampleTo16kHz(audioBuffer);

    // Run Whisper locally — model weights never leave the browser
    const result = await whisperPipeline(float32, {
      chunk_length_s: 30,
      stride_length_s: 5,
      language: "english",
      task: "transcribe",
      return_timestamps: false
    });

    const text = (result.text || "").trim();
    transcript = text;

    if (text) {
      transcriptEl.innerHTML = text;
    } else {
      transcriptEl.innerHTML = '<span class="placeholder">No speech detected. Try recording again.</span>';
    }
  } catch (err) {
    console.error("Whisper transcription error:", err);
    transcriptEl.innerHTML = '<span class="placeholder">Transcription failed. Please try again.</span>';
  } finally {
    isTranscribing = false;
    document.getElementById("micBtn").disabled = false;
    document.getElementById("micHint").textContent = "Click to record again";
    updateProcessBtn();
  }
}

/**
 * Converts an AudioBuffer to a mono 16 kHz Float32Array.
 * If the buffer is already 16 kHz it returns the first channel directly.
 * Otherwise uses OfflineAudioContext to resample.
 */
async function resampleTo16kHz(audioBuffer) {
  const targetRate = 16000;
  const numFrames = audioBuffer.length;
  const srcRate = audioBuffer.sampleRate;

  if (srcRate === targetRate) {
    // Already correct sample rate — just mix down to mono
    const ch0 = audioBuffer.getChannelData(0);
    if (audioBuffer.numberOfChannels === 1) return ch0;
    const ch1 = audioBuffer.getChannelData(1);
    const mono = new Float32Array(ch0.length);
    for (let i = 0; i < ch0.length; i++) mono[i] = (ch0[i] + ch1[i]) / 2;
    return mono;
  }

  // Resample via OfflineAudioContext
  const targetLength = Math.round(numFrames * targetRate / srcRate);
  const offlineCtx = new OfflineAudioContext(1, targetLength, targetRate);
  const offlineSource = offlineCtx.createBufferSource();
  offlineSource.buffer = audioBuffer;
  offlineSource.connect(offlineCtx.destination);
  offlineSource.start();
  const resampled = await offlineCtx.startRendering();
  return resampled.getChannelData(0);
}

function updateTimer() {
  timerSeconds++;
  const m = String(Math.floor(timerSeconds / 60)).padStart(2, "0");
  const s = String(timerSeconds % 60).padStart(2, "0");
  document.getElementById("timerDisplay").textContent = `${m}:${s}`;
}

// ─── Post-Processors ──────────────────────────────────────────────────────

// 1. Strip <think>...</think> blocks that Qwen3 may emit despite /no_think.
function stripThinkTags(raw) {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*/gi, "")
    .trim();
}

// 2. Replace [BOILERPLATE:KEY] tags with guaranteed-correct boilerplate text.
const BOILERPLATE = {
  WCC:
    "All forms, labs, immunizations, and patient concerns reviewed and addressed appropriately. " +
    "Screening questions, past medical history, past social history, medications, and growth chart reviewed. " +
    "Age-appropriate anticipatory guidance reviewed and printed in AVS. Parent questions addressed.",

  ILLNESS:
    "Recommended supportive care with OTC medications as needed. " +
    "Return precautions given including increasing pain, worsening fever, dehydration, new symptoms, " +
    "prolonged symptoms, worsening symptoms, and other concerns. " +
    "Caregiver expressed understanding and agreement with treatment plan.",

  INJURY:
    "Recommended supportive care with Tylenol, Motrin, rest, ice, compression, elevation, and gradual " +
    "return to activity as appropriate. " +
    "Return precautions given including increasing pain, swelling, or failure to improve.",

  OTITIS:
    "Risk of untreated otitis media includes persistent pain and fever, hearing loss, and mastoiditis.",

  STREP:
    "Risk of untreated strep throat includes rheumatic fever and peritonsillar abscess. " +
    "This problem is moderate risk due to pending lab results which may necessitate further pharmacologic management.",

  DEHYDRATION:
    "Patient is at risk for dehydration, which would warrant emergency room care or admission for IV fluids.",

  RESP:
    "Patient is at risk for worsening respiratory distress and clinical deterioration, " +
    "which would need emergency room care or hospital admission.",

  PCMH: "PCMH Reminder"
};

function applyBoilerplate(raw) {
  return raw.replace(/\[BOILERPLATE:([A-Z_]+)\]/g, (match, key) => {
    const text = BOILERPLATE[key];
    if (!text) return "";
    return "\n" + text + "\n";
  });
}

function postProcess(raw) {
  return applyBoilerplate(stripThinkTags(raw));
}

// ─── Process Note ─────────────────────────────────────────────────────────
window.processNote = async function() {
  if (!isModelReady || !engine) return;

  const input = currentTab === "mic"
    ? transcript.trim()
    : document.getElementById("textInput").value.trim();

  if (!input) return;

  const empty = document.getElementById("outputEmpty");
  const content = document.getElementById("outputContent");
  const streaming = document.getElementById("outputStreaming");
  const streamText = document.getElementById("streamText");
  const btnCopy = document.getElementById("btnCopy");

  empty.style.display = "none";
  content.style.display = "none";
  streaming.style.display = "block";
  btnCopy.style.display = "none";
  streamText.textContent = "";

  document.getElementById("btnProcess").disabled = true;

  const systemPrompt = `You are a clinical documentation assistant that converts clinician dictation into concise telegraphic assessment and plan notes.

# OUTPUT FORMAT

For each diagnosis/problem mentioned, present bullets in this order when present:

Diagnosis or Problem Name
- Labs
- Imaging
- Medications with exact doses if stated
- Treatment / plan actions
- Supportive care
- Differential if mentioned
- Conditional plans if mentioned
- Return precautions if mentioned
- Nursing orders if mentioned
- Follow-Up if mentioned

Separate each problem with one blank line.

# STYLE RULES

- Use concise telegraphic bullets only
- No full sentences unless necessary for clarity
- No commentary, explanation, or preamble
- Output ONLY the note
- Do not use markdown formatting — no asterisks, no pound signs, plain text only
- Include only information explicitly stated or clearly implied
- Do not invent diagnoses, medications, labs, imaging, or follow-up
- Preserve clinician wording when reasonable
- Keep diagnoses in order mentioned
- Do not create empty categories or placeholder bullets
- Medication names and doses must match dictation exactly — omit dose if not stated
- Use the explicit diagnosis or condition name as the heading, not presenting symptoms
- If the clinician states a diagnosis, always prefer it over symptom descriptors as the heading

# FORMATTING RULES

- Differentials format:
  Differential includes X, Y, Z

- Return precautions format:
  Return precautions include...

- Follow-up format:
  Follow-Up: ...

# BOILERPLATE TAGS

After all problem blocks, emit the appropriate tag(s) on their own line when the condition is present.
Do not write the boilerplate text yourself — emit only the tag exactly as shown.

- Well child check or health maintenance discussed → [BOILERPLATE:WCC]
- Any illness (infection, virus, fever, etc.) discussed → [BOILERPLATE:ILLNESS]
- Any injury discussed → [BOILERPLATE:INJURY]
- Ear infection (otitis media) discussed → [BOILERPLATE:OTITIS]
- Strep throat or rapid strep test discussed → [BOILERPLATE:STREP]
- Dehydration, vomiting, diarrhea, or decreased urination discussed → [BOILERPLATE:DEHYDRATION]
- Trouble breathing, wheezing, or respiratory distress discussed → [BOILERPLATE:RESP]
- ADHD, weight concern, obesity, or strep throat discussed → [BOILERPLATE:PCMH]

Multiple tags may apply. Each tag goes on its own line after the last problem block.

# EXAMPLES

Dictation: "patient has acute otitis media, plan to treat with amoxicillin 90mg per kg per day divided twice daily, also tylenol motrin and hydration, return precautions for worsening fever or pain, follow up as needed"

Acute Otitis Media
- Amoxicillin 90mg/kg/day divided BID
- Tylenol, Motrin, hydration
- Return precautions include worsening fever, pain, failure to improve
- Follow-Up: PRN
[BOILERPLATE:ILLNESS]
[BOILERPLATE:OTITIS]

---

Dictation: "patient presenting with cough and fever, exam with right lower lobe crackles, diagnosis is community acquired pneumonia, treating with amoxicillin, also supportive care with tylenol motrin and fluids, return precautions for increased work of breathing, follow up as needed"

Community-Acquired Pneumonia, right lower lobe
- Amoxicillin
- Tylenol, Motrin, fluids
- Return precautions include increased work of breathing
- Follow-Up: PRN
[BOILERPLATE:ILLNESS]
[BOILERPLATE:RESP]

---

Dictation: "ADHD combined type, increasing concerta from 18 to 27mg daily, placing counseling referral, follow up in three months"

ADHD, combined
- Concerta increased from 18mg to 27mg PO daily
- Counseling referral placed
- Follow-Up: 3 months
[BOILERPLATE:PCMH]

---

Dictation: "high fever one week, concerned for kawasaki or MIS-C or RMSF, ordering CBC CMP ESR CRP UA and chest xray, giving NS bolus and IVIG, tylenol motrin zofran, if blood pressure drops repeat bolus and consider ICU, vitals every four hours, return precautions for worsening fever new rash or change in mental status, follow up tomorrow or sooner"

Fever
- Differential includes Kawasaki disease, MIS-C, RMSF
- CBC, CMP, ESR, CRP, UA
- Chest XR
- NS bolus, IVIG
- Tylenol, Motrin, Zofran
- If hypotension develops, repeat NS bolus and consider ICU
- Vitals q4hr
- Return precautions include worsening fever, new rash, change in mental status
- Follow-Up: next day or sooner PRN
[BOILERPLATE:ILLNESS]
[BOILERPLATE:DEHYDRATION]

---

Dictation: "well child check, growing and developing well, anticipatory guidance discussed, all questions addressed, follow up in one year"

Well Child Check
- Growing and developing well
- Anticipatory guidance discussed
- Questions addressed
- Follow-Up: 1 year/PRN
[BOILERPLATE:WCC]

---

Dictation: "rash, differential includes ringworm pityriasis rosea and scabies, treating with ketoconazole cream, zyrtec and atarax for itching and sleep, if spreads or fails to improve may consider permethrin, return precautions for worsening, follow up as needed"

Rash
- Differential includes ringworm, pityriasis rosea, scabies
- Ketoconazole cream
- Zyrtec, Atarax for itching and sleep
- If spreads or fails to improve with ketoconazole, consider permethrin
- Return precautions include worsening rash, worsening itch, failure to improve
- Follow-Up: PRN
[BOILERPLATE:ILLNESS]`;

  const userPrompt = `Convert this clinical dictation into structured assessment and plan notes:\n\n${input}\n\n/no_think`;

  try {
    let rawText = "";
    const stream = await engine.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      stream: true,
      temperature: 0.1,
      max_tokens: 1024,
      extra_body: { enable_thinking: false }
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || "";
      rawText += delta;
      streamText.textContent = rawText;
    }

    const processedText = postProcess(rawText);

    streaming.style.display = "none";
    renderOutput(processedText);
    btnCopy.style.display = "flex";
    window._rawOutput = processedText;

  } catch (err) {
    console.error("Generation error:", err);
    streaming.style.display = "none";
    showError("Error generating notes: " + err.message);
  }

  updateProcessBtn();
};

// ─── Render Output ────────────────────────────────────────────────────────
function isBoilerplateParagraph(text) {
  return text.length > 60 && !text.startsWith("-") && !text.startsWith("•");
}

function renderOutput(raw) {
  const content = document.getElementById("outputContent");
  content.style.display = "block";
  content.innerHTML = "";

  const lines = raw.split("\n");
  let currentBlock = null;
  let currentItems = null;
  let blockCount = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const isBullet = trimmed.startsWith("-") || trimmed.startsWith("•");
    const isBoilerplate = isBoilerplateParagraph(trimmed);

    if (isBoilerplate) {
      const bp = document.createElement("div");
      bp.className = "boilerplate-block";
      bp.style.opacity = "0";
      bp.textContent = trimmed;
      content.appendChild(bp);
      currentBlock = null;
      currentItems = null;
      blockCount++;
      requestAnimationFrame(() => { bp.style.opacity = "1"; });

    } else if (!isBullet) {
      const block = document.createElement("div");
      block.className = "problem-block";
      block.style.animationDelay = `${blockCount * 0.08}s`;
      block.style.opacity = "0";

      const title = document.createElement("div");
      title.className = "problem-title";
      title.textContent = trimmed;
      block.appendChild(title);

      const ul = document.createElement("ul");
      ul.className = "problem-items";
      block.appendChild(ul);

      content.appendChild(block);
      currentBlock = block;
      currentItems = ul;
      blockCount++;
      requestAnimationFrame(() => { block.style.opacity = ""; });

    } else if (isBullet && currentItems) {
      const li = document.createElement("li");
      li.textContent = trimmed.replace(/^[-•]\s*/, "");
      currentItems.appendChild(li);
    }
  }

  if (blockCount === 0) {
    content.innerHTML = `<pre style="font-family:var(--font-mono);font-size:0.8rem;line-height:1.8;color:var(--text);padding:1rem;white-space:pre-wrap">${raw}</pre>`;
  }
}

// ─── Copy ─────────────────────────────────────────────────────────────────
window.copyOutput = function() {
  const text = window._rawOutput || "";
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("btnCopy");
    btn.classList.add("copied");
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
    }, 2000);
  });
};

// ─── Clear ────────────────────────────────────────────────────────────────
window.clearAll = function() {
  transcript = "";
  document.getElementById("transcriptText").innerHTML = '<span class="placeholder">Your words will appear here as you speak…</span>';
  document.getElementById("textInput").value = "";
  document.getElementById("outputEmpty").style.display = "flex";
  document.getElementById("outputContent").style.display = "none";
  document.getElementById("outputStreaming").style.display = "none";
  document.getElementById("btnCopy").style.display = "none";

  const area = document.getElementById("outputArea");
  const err = area.querySelector(".error-msg");
  if (err) err.remove();
  area.appendChild(document.getElementById("outputEmpty"));
  area.appendChild(document.getElementById("outputContent"));
  area.appendChild(document.getElementById("outputStreaming"));

  window._rawOutput = "";
  if (isRecording) stopRecording();
  updateProcessBtn();
};

// ─── Text input watcher ───────────────────────────────────────────────────
document.getElementById("textInput").addEventListener("input", updateProcessBtn);

// ─── Boot ─────────────────────────────────────────────────────────────────
// Disable mic button until Whisper is loaded
const micBtnEl = document.getElementById("micBtn");
if (micBtnEl) micBtnEl.disabled = true;
const micHintEl = document.getElementById("micHint");
if (micHintEl) micHintEl.textContent = "Loading local Whisper model…";

// Load both models in parallel for fastest startup
initModel();
initWhisper();
