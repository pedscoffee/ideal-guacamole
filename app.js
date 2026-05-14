// Present — app.js
// WebLLM + Web Speech API for clinical A&P note generation

import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2/dist/transformers.min.js";

env.allowLocalModels = false;

// ─── Default Settings (single source of truth) ────────────────────────────
const DEFAULTS = {
  cleanupPrompt: `You are a medical transcription editor. Your task is to clean up a rough ASR (Automated Speech Recognition) dictation transcript. 
- Fix any spelling errors, phonetic mistakes, and correct medical terminology.
- Remove disfluencies, filler words, and false starts.
- Add proper punctuation and capitalization.
- Do NOT change the clinical meaning, add any new information, or reformat into a list.
- Output ONLY the continuous cleaned transcript paragraph.`,

  mainPrompt: `You are a clinical documentation assistant that converts clinician dictation into concise telegraphic assessment and plan notes.

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

{BOILERPLATE_TRIGGER_LIST}

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

Dictation: "well child check, growing and developing well, anticipatory guidance discussed, all questions addressed, follow up in one year"

Well Child Check
- Growing and developing well
- Anticipatory guidance discussed
- Questions addressed
- Follow-Up: 1 year/PRN
[BOILERPLATE:WCC]`,

  boilerplate: [
    {
      key: "WCC",
      trigger: "Well child check or health maintenance discussed",
      text: "All forms, labs, immunizations, and patient concerns reviewed and addressed appropriately. Screening questions, past medical history, past social history, medications, and growth chart reviewed. Age-appropriate anticipatory guidance reviewed and printed in AVS. Parent questions addressed."
    },
    {
      key: "ILLNESS",
      trigger: "Any illness (infection, virus, fever, etc.) discussed",
      text: "Recommended supportive care with OTC medications as needed. Return precautions given including increasing pain, worsening fever, dehydration, new symptoms, prolonged symptoms, worsening symptoms, and other concerns. Caregiver expressed understanding and agreement with treatment plan."
    },
    {
      key: "INJURY",
      trigger: "Any injury discussed",
      text: "Recommended supportive care with Tylenol, Motrin, rest, ice, compression, elevation, and gradual return to activity as appropriate. Return precautions given including increasing pain, swelling, or failure to improve."
    },
    {
      key: "OTITIS",
      trigger: "Ear infection (otitis media) discussed",
      text: "Risk of untreated otitis media includes persistent pain and fever, hearing loss, and mastoiditis."
    },
    {
      key: "STREP",
      trigger: "Strep throat or rapid strep test discussed",
      text: "Risk of untreated strep throat includes rheumatic fever and peritonsillar abscess. This problem is moderate risk due to pending lab results which may necessitate further pharmacologic management."
    },
    {
      key: "DEHYDRATION",
      trigger: "Dehydration, vomiting, diarrhea, or decreased urination discussed",
      text: "Patient is at risk for dehydration, which would warrant emergency room care or admission for IV fluids."
    },
    {
      key: "RESP",
      trigger: "Trouble breathing, wheezing, or respiratory distress discussed",
      text: "Patient is at risk for worsening respiratory distress and clinical deterioration, which would need emergency room care or hospital admission."
    },
    {
      key: "PCMH",
      trigger: "ADHD, weight concern, obesity, or strep throat discussed",
      text: "PCMH Reminder"
    }
  ]
};

// ─── Settings persistence (localStorage) ─────────────────────────────────
const STORAGE_KEY = "present_settings_v1";

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULTS);
    const saved = JSON.parse(raw);
    return {
      cleanupPrompt: saved.cleanupPrompt ?? DEFAULTS.cleanupPrompt,
      mainPrompt: saved.mainPrompt ?? DEFAULTS.mainPrompt,
      boilerplate: Array.isArray(saved.boilerplate) ? saved.boilerplate : structuredClone(DEFAULTS.boilerplate)
    };
  } catch {
    return structuredClone(DEFAULTS);
  }
}

function saveSettingsToStorage(s) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    return true;
  } catch {
    return false;
  }
}

// Live settings object — all runtime code reads from here
let settings = loadSettings();

// ─── Settings accessors used by processNote ──────────────────────────────
function getBoilerplateMap() {
  const map = {};
  for (const entry of settings.boilerplate) {
    if (entry.key && entry.text) map[entry.key.trim().toUpperCase()] = entry.text.trim();
  }
  return map;
}

function buildBoilerplateTriggerList() {
  return settings.boilerplate
    .filter(e => e.key && e.trigger && e.text)
    .map(e => `- ${e.trigger.trim()} → [BOILERPLATE:${e.key.trim().toUpperCase()}]`)
    .join("\n");
}

function getCleanupSystemPrompt() {
  return settings.cleanupPrompt;
}

function getNoteSystemPrompt() {
  const triggerList = buildBoilerplateTriggerList();
  return settings.mainPrompt.replace("{BOILERPLATE_TRIGGER_LIST}", triggerList);
}

// ─── State ────────────────────────────────────────────────────────────────
let engine = null;
let transcriber = null;
let isLLMReady = false;
let isWhisperReady = false;
let isModelReady = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let transcript = "";
let timerInterval = null;
let timerSeconds = 0;
let currentTab = "mic";

function checkModelsReady() {
  if (isLLMReady && isWhisperReady) {
    isModelReady = true;
    setStatus("ready", "Models ready");
    showProgress(false);
    updateProcessBtn();
  }
}

// ─── Model Setup ──────────────────────────────────────────────────────────
const MODEL_ID = "Qwen3-4B-q4f16_1-MLC";

async function initModel() {
  setStatus("loading", "Initializing models…");
  showProgress(true, "Downloading models (LLM ~2.3GB, Whisper ~75MB)…", 0);

  pipeline("automatic-speech-recognition", "Xenova/whisper-tiny.en", {
    device: "wasm",
    progress_callback: (progress) => {
      if (progress.status === 'progress' && !isLLMReady) {
        showProgress(true, `Loading Whisper…`, Math.round(progress.progress || 0));
      }
    }
  }).then(t => {
    transcriber = t;
    isWhisperReady = true;
    checkModelsReady();
  }).catch(err => {
    console.error("Whisper init failed:", err);
    showError("Failed to load Whisper model.");
  });

  try {
    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (progress) => {
        const pct = Math.round((progress.progress || 0) * 100);
        showProgress(true, progress.text || "Loading LLM…", pct);
      }
    });
    isLLMReady = true;
    checkModelsReady();
  } catch (err) {
    console.error("Model init failed:", err);
    setStatus("error", "Model failed to load");
    showProgress(false);
    showError("Failed to load the AI model. Please check your browser supports WebGPU (Chrome 113+ recommended) and reload.");
  }
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
    ? (document.getElementById("transcriptText").value || transcript).trim().length > 20
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

// ─── Speech Recognition ───────────────────────────────────────────────────
async function setupAudioRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];
      await transcribeAudio(audioBlob);
    };
  } catch (err) {
    console.error("Microphone access denied:", err);
    document.getElementById("micHint").textContent = "Microphone access denied. Please check permissions.";
    document.getElementById("micBtn").disabled = true;
  }
}

async function transcribeAudio(blob) {
  if (!transcriber) return;
  const ta = document.getElementById("transcriptText");
  ta.value = "";
  ta.placeholder = "Transcribing securely in browser…";
  ta.readOnly = true;
  ta.classList.remove("is-editable");
  document.getElementById("transcriptLabel").textContent = "Transcript";

  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const audioData = audioBuffer.getChannelData(0);

    const result = await transcriber(audioData);
    transcript = result.text.trim();

    if (transcript) {
      ta.value = transcript;
      ta.readOnly = false;
      ta.classList.add("is-editable");
      document.getElementById("transcriptLabel").textContent = "Transcript — editable";
      ta.oninput = () => { transcript = ta.value; updateProcessBtn(); };
    } else {
      ta.value = "";
      ta.placeholder = "No speech detected.";
    }

    document.getElementById("micHint").textContent = "Click to begin recording";
    updateProcessBtn();
  } catch (err) {
    console.error("Transcription error:", err);
    ta.value = "";
    ta.placeholder = "Failed to transcribe audio.";
    ta.readOnly = true;
    ta.classList.remove("is-editable");
    document.getElementById("micHint").textContent = "Click to begin recording";
  }
}

window.toggleRecording = async function() {
  if (isRecording) stopRecording();
  else await startRecording();
};

async function startRecording() {
  if (!mediaRecorder) {
    await setupAudioRecording();
    if (!mediaRecorder) return;
  }
  audioChunks = [];
  mediaRecorder.start();
  isRecording = true;
  document.getElementById("micVisualizer").classList.add("recording");
  document.getElementById("micBtn").style.cssText = "";
  document.getElementById("micHint").textContent = "Listening… speak your assessment and plan";
  document.getElementById("recordingTimer").style.display = "flex";
  const ta = document.getElementById("transcriptText");
  ta.value = "";
  ta.placeholder = "Recording in progress… Processing starts when you stop.";
  ta.readOnly = true;
  ta.classList.remove("is-editable");
  document.getElementById("transcriptLabel").textContent = "Transcript";
  timerSeconds = 0;
  updateTimer();
  timerInterval = setInterval(updateTimer, 1000);
}

function stopRecording() {
  isRecording = false;
  mediaRecorder.stop();
  document.getElementById("micVisualizer").classList.remove("recording");
  document.getElementById("micHint").textContent = "Processing audio locally...";
  document.getElementById("recordingTimer").style.display = "none";
  clearInterval(timerInterval);
}

function updateTimer() {
  timerSeconds++;
  const m = String(Math.floor(timerSeconds / 60)).padStart(2, "0");
  const s = String(timerSeconds % 60).padStart(2, "0");
  document.getElementById("timerDisplay").textContent = `${m}:${s}`;
}

// ─── Post-Processors ──────────────────────────────────────────────────────

// Strip <think>...</think> blocks that Qwen3 may emit despite /no_think.
function stripThinkTags(raw) {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*/gi, "")
    .trim();
}

// Replace [BOILERPLATE:KEY] tags using current live settings.
function applyBoilerplate(raw) {
  const bpMap = getBoilerplateMap();
  return raw.replace(/\[BOILERPLATE:([A-Z_]+)\]/g, (match, key) => {
    const text = bpMap[key];
    if (!text) return "";
    return "\n" + text + "\n";
  });
}

function postProcess(raw) {
  return applyBoilerplate(stripThinkTags(raw));
}

// ─── Autocopy helper ──────────────────────────────────────────────────────
function autoCopyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    showAutocopyToast("✓ Auto-copied to clipboard");
  }).catch(() => {});
}

function showAutocopyToast(msg) {
  const toast = document.getElementById("autocopyToast");
  toast.textContent = msg;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 2500);
}

// ─── Get current output text (handles edits made to contenteditable) ──────
function getCurrentOutputText() {
  const content = document.getElementById("outputContent");
  const lines = [];
  for (const node of content.children) {
    if (node.classList.contains("problem-block")) {
      const title = node.querySelector(".problem-title");
      if (title) lines.push(title.textContent.trim());
      const items = node.querySelectorAll(".problem-items li");
      items.forEach(li => lines.push("- " + li.textContent.trim()));
      lines.push("");
    } else if (node.classList.contains("boilerplate-block")) {
      lines.push(node.textContent.trim());
      lines.push("");
    } else {
      const t = node.textContent.trim();
      if (t) lines.push(t);
    }
  }
  return lines.join("\n").trim();
}

// ─── Process Note ─────────────────────────────────────────────────────────
window.processNote = async function() {
  if (!isModelReady || !engine) return;

  const input = currentTab === "mic"
    ? document.getElementById("transcriptText").value.trim()
    : document.getElementById("textInput").value.trim();

  if (!input) return;

  const empty = document.getElementById("outputEmpty");
  const content = document.getElementById("outputContent");
  const streaming = document.getElementById("outputStreaming");
  const streamText = document.getElementById("streamText");
  const btnCopy = document.getElementById("btnCopy");
  const btnCopyGroup = document.getElementById("btnCopyGroup");
  const editHint = document.getElementById("editHint");
  const thinkLabel = document.querySelector(".thinking-label");

  empty.style.display = "none";
  content.style.display = "none";
  content.contentEditable = "false";
  streaming.style.display = "block";
  btnCopy.style.display = "none";
  if (btnCopyGroup) btnCopyGroup.style.display = "none";
  editHint.style.display = "none";
  streamText.textContent = "";
  document.getElementById("btnProcess").disabled = true;

  try {
    // --- PASS 1: Cleanup transcript ---
    thinkLabel.innerHTML = '<span class="think-dot"></span>Cleaning transcript…';

    const cleanupResponse = await engine.chat.completions.create({
      messages: [
        { role: "system", content: getCleanupSystemPrompt() },
        { role: "user", content: input + "\n\n/no_think" }
      ],
      stream: false,
      temperature: 0.1,
      max_tokens: 512,
      extra_body: { enable_thinking: false }
    });

    let cleanedInput = stripThinkTags(cleanupResponse.choices[0]?.message?.content || input);
    if (!cleanedInput.trim()) cleanedInput = input;

    if (currentTab === "mic") {
      transcript = cleanedInput;
      const ta = document.getElementById("transcriptText");
      ta.value = cleanedInput;
      ta.readOnly = false;
      ta.classList.add("is-editable");
      document.getElementById("transcriptLabel").textContent = "Transcript — editable";
      ta.oninput = () => { transcript = ta.value; updateProcessBtn(); };
    } else {
      document.getElementById("textInput").value = cleanedInput;
    }

    // --- PASS 2: Generate Notes ---
    thinkLabel.innerHTML = '<span class="think-dot"></span>Generating notes…';
    streamText.textContent = "";

    const userPrompt = `Convert this clinical dictation into structured assessment and plan notes:\n\n${cleanedInput}\n\n/no_think`;

    let rawText = "";
    const stream = await engine.chat.completions.create({
      messages: [
        { role: "system", content: getNoteSystemPrompt() },
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
    if (btnCopyGroup) btnCopyGroup.style.display = "flex";
    btnCopy.style.display = "flex";
    editHint.style.display = "flex";
    window._rawOutput = processedText;

    setTimeout(() => { content.contentEditable = "true"; }, 200);
    setTimeout(() => { autoCopyToClipboard(processedText); }, 400);

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
  let problemCount = 0;

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
      problemCount++;
      const pIdx = problemCount;

      const block = document.createElement("div");
      block.className = "problem-block";
      block.dataset.problemIndex = pIdx;
      block.style.animationDelay = `${blockCount * 0.08}s`;
      block.style.opacity = "0";

      const titleRow = document.createElement("div");
      titleRow.className = "problem-title-row";

      const title = document.createElement("div");
      title.className = "problem-title";
      title.textContent = trimmed;
      titleRow.appendChild(title);

      const copyBtn = document.createElement("button");
      copyBtn.className = "btn-copy-problem";
      copyBtn.setAttribute("aria-label", "Copy this problem block");
      copyBtn.title = "Copy this problem";
      copyBtn.dataset.problemIndex = pIdx;
      copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
      copyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        copyProblemBlock(block, copyBtn);
      });
      titleRow.appendChild(copyBtn);

      block.appendChild(titleRow);

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

  // Update the dropdown after render
  updateCopyDropdown();
}

// ─── Copy ─────────────────────────────────────────────────────────────────

// Shared flash-feedback for any copy button
function flashCopied(btn, resetHTML) {
  btn.classList.add("copied");
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><polyline points="20 6 9 17 4 12"/></svg>`;
  setTimeout(() => {
    btn.classList.remove("copied");
    btn.innerHTML = resetHTML;
  }, 1800);
}

// Copy all problems (used by main "Copy All" button)
window.copyOutput = function() {
  const text = getCurrentOutputText() || window._rawOutput || "";
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("btnCopy");
    btn.classList.add("copied");
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy All`;
    }, 2000);
  });
};

// Extract plain text from a single problem-block element
function getProblemBlockText(block) {
  const lines = [];
  const title = block.querySelector(".problem-title");
  if (title) lines.push(title.textContent.trim());
  const items = block.querySelectorAll(".problem-items li");
  items.forEach(li => lines.push("- " + li.textContent.trim()));
  return lines.join("\n");
}

// Copy a single problem block — called from inline copy buttons
function copyProblemBlock(block, btn) {
  const text = getProblemBlockText(block);
  const resetHTML = btn.innerHTML;
  navigator.clipboard.writeText(text).then(() => {
    flashCopied(btn, resetHTML);
    showAutocopyToast("✓ Problem copied");
  }).catch(() => {});
}

// Copy a single problem by index (1-based) — called from dropdown
window.copyProblemByIndex = function(idx) {
  const block = document.querySelector(`.problem-block[data-problem-index="${idx}"]`);
  if (!block) return;
  const text = getProblemBlockText(block);
  navigator.clipboard.writeText(text).then(() => {
    showAutocopyToast("✓ Problem " + idx + " copied");
    closeCopyDropdown();
  }).catch(() => {});
};

// Build/refresh the dropdown with one entry per problem block
function updateCopyDropdown() {
  const dropdown = document.getElementById("copyDropdown");
  if (!dropdown) return;
  dropdown.innerHTML = "";

  const blocks = document.querySelectorAll(".problem-block");

  // "Copy All" option at top
  const allOpt = document.createElement("button");
  allOpt.className = "copy-dropdown-item copy-dropdown-all";
  allOpt.setAttribute("role", "option");
  allOpt.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy All`;
  allOpt.addEventListener("click", () => { copyOutput(); closeCopyDropdown(); });
  dropdown.appendChild(allOpt);

  if (blocks.length > 1) {
    const divider = document.createElement("div");
    divider.className = "copy-dropdown-divider";
    dropdown.appendChild(divider);

    blocks.forEach((block, i) => {
      const titleEl = block.querySelector(".problem-title");
      const label = titleEl ? titleEl.textContent.trim() : `Problem ${i + 1}`;
      const shortLabel = label.length > 34 ? label.slice(0, 32) + "…" : label;

      const opt = document.createElement("button");
      opt.className = "copy-dropdown-item";
      opt.setAttribute("role", "option");
      opt.dataset.problemIndex = i + 1;
      opt.innerHTML = `<span class="copy-dropdown-num">${i + 1}</span>${shortLabel}`;
      opt.addEventListener("click", () => window.copyProblemByIndex(i + 1));
      dropdown.appendChild(opt);
    });
  }
}

// Toggle dropdown visibility
window.toggleCopyDropdown = function(e) {
  e.stopPropagation();
  const dropdown = document.getElementById("copyDropdown");
  const chevron = document.getElementById("btnCopyChevron");
  const isOpen = dropdown.classList.contains("open");
  if (isOpen) {
    closeCopyDropdown();
  } else {
    dropdown.classList.add("open");
    chevron.setAttribute("aria-expanded", "true");
  }
};

function closeCopyDropdown() {
  const dropdown = document.getElementById("copyDropdown");
  const chevron = document.getElementById("btnCopyChevron");
  if (dropdown) dropdown.classList.remove("open");
  if (chevron) chevron.setAttribute("aria-expanded", "false");
}

// Close dropdown when clicking outside
document.addEventListener("click", () => closeCopyDropdown());

// ─── Clear ────────────────────────────────────────────────────────────────
window.clearAll = function() {
  transcript = "";
  const ta = document.getElementById("transcriptText");
  ta.value = "";
  ta.placeholder = "Your words will appear here as you speak…";
  ta.readOnly = true;
  ta.classList.remove("is-editable");
  ta.oninput = null;
  document.getElementById("transcriptLabel").textContent = "Transcript";
  document.getElementById("textInput").value = "";
  document.getElementById("outputEmpty").style.display = "flex";

  const content = document.getElementById("outputContent");
  content.style.display = "none";
  content.contentEditable = "false";
  content.innerHTML = "";

  document.getElementById("outputStreaming").style.display = "none";
  document.getElementById("btnCopy").style.display = "none";
  const _cg2 = document.getElementById("btnCopyGroup"); if (_cg2) _cg2.style.display = "none";
  document.getElementById("editHint").style.display = "none";

  const area = document.getElementById("outputArea");
  const err = area.querySelector(".error-msg");
  if (err) err.remove();
  area.appendChild(document.getElementById("outputEmpty"));
  area.appendChild(content);
  area.appendChild(document.getElementById("outputStreaming"));

  window._rawOutput = "";
  if (isRecording) {
    const prevOnStop = mediaRecorder.onstop;
    mediaRecorder.onstop = null;
    stopRecording();
    mediaRecorder.onstop = prevOnStop;
  }
  updateProcessBtn();
};

// ─── Text input watcher ───────────────────────────────────────────────────
document.getElementById("textInput").addEventListener("input", updateProcessBtn);

// ═══════════════════════════════════════════════════════════════════════════
// SETTINGS DRAWER
// ═══════════════════════════════════════════════════════════════════════════

window.openSettings = function() {
  populateSettingsUI();
  document.getElementById("settingsDrawer").classList.add("open");
  document.getElementById("settingsOverlay").classList.add("open");
  document.body.style.overflow = "hidden";
};

window.closeSettings = function() {
  document.getElementById("settingsDrawer").classList.remove("open");
  document.getElementById("settingsOverlay").classList.remove("open");
  document.body.style.overflow = "";
};

function populateSettingsUI() {
  document.getElementById("settingCleanupPrompt").value = settings.cleanupPrompt;
  document.getElementById("settingMainPrompt").value = settings.mainPrompt;
  renderBoilerplateList();
}

function renderBoilerplateList() {
  const container = document.getElementById("boilerplateList");
  container.innerHTML = "";
  settings.boilerplate.forEach((entry, idx) => {
    container.appendChild(createBoilerplateEntryEl(entry, idx));
  });
}

function escHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function createBoilerplateEntryEl(entry, idx) {
  const div = document.createElement("div");
  div.className = "bp-entry";
  div.dataset.idx = idx;

  div.innerHTML = `
    <div class="bp-entry-header" onclick="toggleBpEntry(this)">
      <span class="bp-entry-key-badge">[BOILERPLATE:<input
        class="bp-entry-key-input"
        type="text"
        value="${escHtml(entry.key)}"
        placeholder="KEY"
        maxlength="30"
        spellcheck="false"
        onclick="event.stopPropagation()"
        oninput="updateBpKey(${idx}, this.value)"
      />]</span>
      <span class="bp-entry-toggle">
        <svg class="bp-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
      </span>
    </div>
    <div class="bp-entry-body">
      <div>
        <div class="bp-field-label">Trigger phrase(s)</div>
        <textarea
          class="bp-entry-trigger"
          rows="2"
          placeholder="e.g. Well child check or health maintenance discussed"
          oninput="updateBpField(${idx}, 'trigger', this.value)"
        >${escHtml(entry.trigger)}</textarea>
      </div>
      <div>
        <div class="bp-field-label">Boilerplate text</div>
        <textarea
          class="bp-entry-text"
          rows="4"
          oninput="updateBpField(${idx}, 'text', this.value)"
        >${escHtml(entry.text)}</textarea>
      </div>
      <div class="bp-entry-footer">
        <button class="bp-btn-delete" onclick="deleteBpEntry(${idx})">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>
          Delete
        </button>
      </div>
    </div>`;
  return div;
}

window.toggleBpEntry = function(header) {
  header.closest(".bp-entry").classList.toggle("expanded");
};

window.updateBpKey = function(idx, val) {
  settings.boilerplate[idx].key = val.toUpperCase().replace(/[^A-Z0-9_]/g, "");
};

window.updateBpField = function(idx, field, val) {
  settings.boilerplate[idx][field] = val;
};

window.deleteBpEntry = function(idx) {
  settings.boilerplate.splice(idx, 1);
  renderBoilerplateList();
};

window.addBoilerplateEntry = function() {
  settings.boilerplate.push({ key: "", trigger: "", text: "" });
  renderBoilerplateList();
  const list = document.getElementById("boilerplateList");
  const last = list.lastElementChild;
  if (last) {
    last.classList.add("expanded");
    last.querySelector(".bp-entry-key-input")?.focus();
  }
};

window.saveSettings = function() {
  // Read prompts back from textareas (boilerplate already live via oninput)
  settings.cleanupPrompt = document.getElementById("settingCleanupPrompt").value;
  settings.mainPrompt = document.getElementById("settingMainPrompt").value;

  const ok = saveSettingsToStorage(settings);
  const status = document.getElementById("settingsSaveStatus");
  status.textContent = ok ? "✓ Saved" : "⚠ Could not persist (storage blocked)";
  status.classList.add("visible");
  setTimeout(() => status.classList.remove("visible"), 2500);
};

window.resetSettings = function() {
  if (!confirm("Reset all settings to defaults? This cannot be undone.")) return;
  settings = structuredClone(DEFAULTS);
  saveSettingsToStorage(settings);
  populateSettingsUI();
  const status = document.getElementById("settingsSaveStatus");
  status.textContent = "✓ Reset to defaults";
  status.classList.add("visible");
  setTimeout(() => status.classList.remove("visible"), 2500);
};

// Close drawer on Escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const drawer = document.getElementById("settingsDrawer");
    if (drawer.classList.contains("open")) closeSettings();
  }
});

// ─── Boot ─────────────────────────────────────────────────────────────────
initModel();
