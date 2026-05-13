// Present — app.js
// 100% local: WebLLM (Qwen3) for note generation + Transformers.js Whisper for STT
// No data ever leaves the device. No Web Speech API. No external APIs.

import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// ─── State ────────────────────────────────────────────────────────────────
let engine       = null;
let isLLMReady   = false;
let isWhisperReady = false;
let isRecording  = false;
let mediaRecorder = null;
let audioChunks  = [];
let whisperWorker = null;
let transcript   = "";
let timerInterval = null;
let timerSeconds = 0;
let currentTab   = "mic";
let audioContext = null;

// ─── Dual-model status tracking ───────────────────────────────────────────
// We load both models; the app becomes usable once BOTH are ready.
let llmProgress     = 0;
let whisperProgress = 0;

function updateCombinedStatus() {
  if (isLLMReady && isWhisperReady) {
    setStatus("ready", "All models ready — fully local");
    showProgress(false);
    updateProcessBtn();
  } else if (!isLLMReady && !isWhisperReady) {
    const avg = Math.round((llmProgress + whisperProgress) / 2);
    showProgress(true, `Loading models… LLM ${llmProgress}% | Whisper ${whisperProgress}%`, avg);
  } else if (!isLLMReady) {
    showProgress(true, `Loading LLM… ${llmProgress}%`, llmProgress);
  } else {
    showProgress(true, `Loading Whisper… ${whisperProgress}%`, whisperProgress);
  }
}

// ─── Whisper Worker Setup ─────────────────────────────────────────────────
function initWhisper() {
  setStatus("loading", "Loading Whisper STT…");
  whisperWorker = new Worker(
    new URL("./whisper-worker.js", import.meta.url),
    { type: "module" }
  );

  whisperWorker.onmessage = (e) => {
    const { type, data, text, message } = e.data;

    if (type === "progress") {
      if (data && data.progress != null) {
        whisperProgress = Math.round(data.progress * 100);
        updateCombinedStatus();
      }
    }

    if (type === "ready") {
      isWhisperReady = true;
      updateCombinedStatus();
    }

    if (type === "transcript") {
      setTranscript(text);
      setMicHint("Recording complete — click Generate Notes");
      stopRecordingUI();
    }

    if (type === "error") {
      console.error("Whisper worker error:", message);
      stopRecordingUI();
      setMicHint("Transcription error — try again");
    }
  };

  whisperWorker.postMessage({ type: "load" });
}

// ─── LLM Setup ────────────────────────────────────────────────────────────
const MODEL_ID = "Qwen3-4B-q4f16_1-MLC";

async function initModel() {
  try {
    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (progress) => {
        llmProgress = Math.round((progress.progress || 0) * 100);
        updateCombinedStatus();
      }
    });
    isLLMReady = true;
    updateCombinedStatus();
    updateProcessBtn();
  } catch (err) {
    console.error("LLM init failed:", err);
    setStatus("error", "LLM failed to load");
    showProgress(false);
    showError("Failed to load the AI model. Please check your browser supports WebGPU (Chrome 113+ recommended) and reload.");
  }
}

// ─── UI helpers ───────────────────────────────────────────────────────────
function setStatus(state, text) {
  const dot   = document.getElementById("statusDot");
  const label = document.getElementById("statusText");
  dot.className = "status-dot " + state;
  label.textContent = text;
}

function showProgress(visible, label = "", pct = 0) {
  const wrap = document.getElementById("progressWrap");
  const bar  = document.getElementById("progressBar");
  const lbl  = document.getElementById("progressLabel");
  if (visible) {
    wrap.style.display = "block";
    bar.style.width    = pct + "%";
    lbl.textContent    = label;
  } else {
    wrap.style.display = "none";
  }
}

function showError(msg) {
  const area      = document.getElementById("outputArea");
  const empty     = document.getElementById("outputEmpty");
  const content   = document.getElementById("outputContent");
  const streaming = document.getElementById("outputStreaming");
  empty.style.display     = "none";
  content.style.display   = "none";
  streaming.style.display = "none";
  area.innerHTML = `<div class="error-msg">${msg}</div>`;
}

function updateProcessBtn() {
  const btn = document.getElementById("btnProcess");
  const hasContent = currentTab === "mic"
    ? transcript.trim().length > 20
    : document.getElementById("textInput").value.trim().length > 20;
  btn.disabled = !isLLMReady || !hasContent;
}

function setTranscript(text) {
  transcript = text;
  const el = document.getElementById("transcriptText");
  el.innerHTML = "";
  el.textContent = text || "";
  if (!text) {
    const ph = document.createElement("span");
    ph.className = "placeholder";
    ph.textContent = "Your words will appear here as you speak…";
    el.appendChild(ph);
  }
  updateProcessBtn();
}

function setMicHint(text) {
  document.getElementById("micHint").textContent = text;
}

// ─── Audio Recording → Whisper ────────────────────────────────────────────
window.toggleRecording = async function() {
  if (!isWhisperReady) {
    setMicHint("Whisper model still loading…");
    return;
  }
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
};

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    audioContext = new AudioContext({ sampleRate: 16000 });
    audioChunks  = [];

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      // Decode audio blob → Float32Array at 16kHz for Whisper
      const blob        = new Blob(audioChunks, { type: mediaRecorder.mimeType });
      const arrayBuffer = await blob.arrayBuffer();

      try {
        const decoded = await audioContext.decodeAudioData(arrayBuffer);
        // Resample to 16kHz mono Float32Array
        const offlineCtx = new OfflineAudioContext(1, decoded.duration * 16000, 16000);
        const source = offlineCtx.createBufferSource();
        source.buffer = decoded;
        source.connect(offlineCtx.destination);
        source.start(0);
        const resampled = await offlineCtx.startRendering();
        const float32   = resampled.getChannelData(0);

        setMicHint("Transcribing audio on-device…");
        // Transfer ownership of the buffer for zero-copy
        whisperWorker.postMessage(
          { type: "transcribe", audio: float32, sampleRate: 16000 },
          [float32.buffer]
        );
      } catch (err) {
        console.error("Audio decode error:", err);
        setMicHint("Audio decode failed — try again");
        stopRecordingUI();
      }

      // Stop all tracks to release microphone
      stream.getTracks().forEach(t => t.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    startTimer();

    const vis = document.getElementById("micVisualizer");
    vis.classList.add("recording");
    document.getElementById("micBtn").style.borderColor = "";
    document.getElementById("recordingTimer").style.display = "flex";
    setMicHint("Recording… click to stop");
    updatePrivacyIndicator("recording");

  } catch (err) {
    console.error("Mic access error:", err);
    setMicHint("Microphone access denied");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  stopTimer();
  updatePrivacyIndicator("idle");
}

function stopRecordingUI() {
  isRecording = false;
  stopTimer();
  const vis = document.getElementById("micVisualizer");
  vis.classList.remove("recording");
  document.getElementById("recordingTimer").style.display = "none";
  updatePrivacyIndicator("idle");
  updateProcessBtn();
}

// ─── Timer ────────────────────────────────────────────────────────────────
function startTimer() {
  timerSeconds = 0;
  timerInterval = setInterval(() => {
    timerSeconds++;
    const m = String(Math.floor(timerSeconds / 60)).padStart(2, "0");
    const s = String(timerSeconds % 60).padStart(2, "0");
    document.getElementById("timerDisplay").textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
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

// ─── Boilerplate Library ──────────────────────────────────────────────────
const BOILERPLATE = {
  WCC:         "All results reviewed and discussed with family. Anticipatory guidance provided. All questions answered. Family verbalized understanding.",
  ILLNESS:     "Illness anticipatory guidance provided including duration of illness, fever management, fluid intake, and activity restrictions. Family verbalized understanding.",
  INJURY:      "Injury anticipatory guidance provided including activity restrictions, wound care, and safety precautions. Family verbalized understanding.",
  OTITIS:      "Otitis media anticipatory guidance provided including expected duration, pain management, hearing precautions, and indications for return. Family verbalized understanding.",
  STREP:       "Strep anticipatory guidance provided including expected duration, medication compliance, and indications for return. Family verbalized understanding.",
  DEHYDRATION: "Dehydration anticipatory guidance provided including oral rehydration therapy, signs of worsening dehydration, and urine output monitoring. Family verbalized understanding.",
  RESP:        "Respiratory anticipatory guidance provided including home nebulizer use, signs of respiratory distress, and indications for return. Family verbalized understanding.",
  PCMH:        "Patient-centered medical home responsibilities discussed including care coordination, chronic disease management, and preventive services. Family verbalized understanding.",
};

// ─── Post-process: strip think tags + inject boilerplate ──────────────────
function stripThinkTags(raw) {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*/gi, "")   // unclosed tag: drop everything after
    .trim();
}

function applyBoilerplate(text) {
  let out = text;
  for (const [key, value] of Object.entries(BOILERPLATE)) {
    const tag = `[BOILERPLATE:${key}]`;
    out = out.replace(new RegExp(escapeRegex(tag), "g"), value);
  }
  // Strip any unrecognized boilerplate tags
  out = out.replace(/\[BOILERPLATE:[A-Z]+\]/g, "");
  return out.trim();
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function postProcess(raw) {
  return applyBoilerplate(stripThinkTags(raw));
}

// ─── New Patient (clear) ──────────────────────────────────────────────────
window.clearAll = function() {
  // Clear all in-memory data — nothing persists between patients
  transcript = "";
  window._rawOutput = "";

  setTranscript("");
  document.getElementById("textInput").value = "";

  const outputArea = document.getElementById("outputArea");
  outputArea.innerHTML = `
    <div class="output-empty" id="outputEmpty">
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
      </div>
      <p>Structured notes will appear here after processing</p>
    </div>
    <div class="output-content" id="outputContent" style="display:none"></div>
    <div class="output-streaming" id="outputStreaming" style="display:none">
      <div class="thinking-label">
        <span class="think-dot"></span>
        Generating notes…
      </div>
      <div class="stream-text" id="streamText"></div>
    </div>`;

  document.getElementById("btnCopy").style.display = "none";
  setMicHint("Click to begin recording");
  stopRecordingUI();
  updateProcessBtn();

  // Flash confirmation
  showClearFlash();
  updatePrivacyIndicator("idle");
};

function showClearFlash() {
  const flash = document.getElementById("clearFlash");
  if (!flash) return;
  flash.classList.add("visible");
  setTimeout(() => flash.classList.remove("visible"), 2000);
}

// ─── Privacy Indicator ────────────────────────────────────────────────────
function updatePrivacyIndicator(state) {
  const dot    = document.getElementById("privacyDot");
  const label  = document.getElementById("privacyLabel");
  if (!dot || !label) return;

  if (state === "recording") {
    dot.className   = "privacy-dot amber";
    label.textContent = "🎙 Mic active — processing on-device";
  } else if (state === "generating") {
    dot.className   = "privacy-dot amber";
    label.textContent = "⚙ Generating — LLM on-device";
  } else {
    dot.className   = "privacy-dot green";
    label.textContent = "🔒 No patient data stored or transmitted";
  }
}

// ─── Privacy Modal ────────────────────────────────────────────────────────
window.openPrivacyModal = function() {
  document.getElementById("privacyModal").classList.add("open");
};
window.closePrivacyModal = function() {
  document.getElementById("privacyModal").classList.remove("open");
};
// Close on backdrop click
document.addEventListener("click", (e) => {
  const modal = document.getElementById("privacyModal");
  if (modal && modal.classList.contains("open") && e.target === modal) {
    modal.classList.remove("open");
  }
});

// ─── Generate Notes ───────────────────────────────────────────────────────
window.processNote = async function() {
  const input = currentTab === "mic"
    ? transcript.trim()
    : document.getElementById("textInput").value.trim();

  if (!input || !isLLMReady) return;

  const btnCopy   = document.getElementById("btnCopy");
  const empty     = document.getElementById("outputEmpty");
  const content   = document.getElementById("outputContent");
  const streaming = document.getElementById("outputStreaming");
  const streamText = document.getElementById("streamText");

  empty.style.display     = "none";
  content.style.display   = "none";
  streaming.style.display = "block";
  btnCopy.style.display   = "none";
  streamText.textContent  = "";

  document.getElementById("btnProcess").disabled = true;
  updatePrivacyIndicator("generating");

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
        { role: "user",   content: userPrompt }
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

  updatePrivacyIndicator("idle");
  updateProcessBtn();
};

// ─── Render Output ────────────────────────────────────────────────────────
function isBoilerplateParagraph(text) {
  return text.length > 60 && !text.startsWith("-") && !text.startsWith("•");
}

function renderOutput(raw) {
  const content = document.getElementById("outputContent");
  content.style.display = "block";
  content.innerHTML     = "";

  const lines = raw.split("\n");
  let currentItems = null;
  let blockCount   = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const isBullet     = trimmed.startsWith("-") || trimmed.startsWith("•");
    const isBoilerplate = isBoilerplateParagraph(trimmed);

    if (isBoilerplate) {
      const bp = document.createElement("div");
      bp.className     = "boilerplate-block";
      bp.style.opacity = "0";
      bp.textContent   = trimmed;
      content.appendChild(bp);
      currentItems = null;
      blockCount++;
      requestAnimationFrame(() => { bp.style.opacity = "1"; });

    } else if (!isBullet) {
      const block = document.createElement("div");
      block.className           = "problem-block";
      block.style.animationDelay = `${blockCount * 0.08}s`;
      block.style.opacity       = "0";

      const title = document.createElement("div");
      title.className = "problem-title";
      title.textContent = trimmed;
      block.appendChild(title);

      const ul = document.createElement("ul");
      ul.className = "problem-items";
      block.appendChild(ul);

      content.appendChild(block);
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

// ─── Init ─────────────────────────────────────────────────────────────────
updatePrivacyIndicator("idle");
initWhisper();
initModel();
