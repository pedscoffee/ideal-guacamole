// Present — app.js
// WebLLM + Transformers.js Whisper for clinical A&P note generation

import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2/dist/transformers.min.js";

env.allowLocalModels = false;

// ─── Model Catalogs ───────────────────────────────────────────────────────
const LLM_MODELS = {
  "gemma-2-2b-it-q4f32_1-MLC":         { label: "Gemma 2 2B (Fast)",            size: "~1.3 GB" },
  "Phi-3.5-mini-instruct-q4f16_1-MLC": { label: "Phi 3.5 Mini (Balanced-Light)", size: "~2.2 GB" },
  "Qwen3-4B-q4f16_1-MLC":              { label: "Qwen3 4B (Balanced)",           size: "~2.3 GB" },
  "Llama-3-8B-Instruct-q4f16_1-MLC":   { label: "Llama 3 8B (Best)",             size: "~4.5 GB" },
};

const WHISPER_MODELS = {
  "Xenova/whisper-tiny.en":   { label: "Whisper Tiny",   size: "~75 MB",  device: "wasm" },
  "Xenova/whisper-base.en":   { label: "Whisper Base",   size: "~145 MB", device: "wasm" },
  "Xenova/whisper-small.en":  { label: "Whisper Small",  size: "~488 MB", device: "wasm" },
  "Xenova/whisper-medium.en": { label: "Whisper Medium", size: "~1.5 GB", device: "wasm" },
};

// ─── Default Settings ─────────────────────────────────────────────────────
const DEFAULTS = {
  llmModel: "Qwen3-4B-q4f16_1-MLC",
  whisperModel: "Xenova/whisper-small.en",
  ephemeralMode: true,
  deidentifyInput: false,
  autoCopyOutput: false,
  termVocabulary: [
    "amoxicillin",
    "rocephin",
    "augmentin",
    "clindamycin",
    "keflex",
    "cefprozil",
    "Tylenol",
    "Motrin",
    "Pedialyte"
  ],
  cleanupPrompt: `You are a medical transcription editor. Your task is to clean up a rough ASR (Automated Speech Recognition) dictation transcript. \n- Fix any spelling errors, phonetic mistakes, and correct medical terminology.\n- Remove disfluencies, filler words, and false starts.\n- Add proper punctuation and capitalization.\n- Do NOT change the clinical meaning, add any new information, or reformat into a list.\n- Output ONLY the continuous cleaned transcript paragraph.`,
  mainPrompt: `You are a clinical documentation assistant that converts clinician dictation into concise telegraphic assessment and plan notes.\n\n# OUTPUT FORMAT\n\nFor each diagnosis/problem mentioned, present bullets in this order when present:\n\nDiagnosis or Problem Name\n- Labs\n- Imaging\n- Medications with exact doses if stated\n- Treatment / plan actions\n- Supportive care\n- Differential if mentioned\n- Conditional plans if mentioned\n- Return precautions if mentioned\n- Nursing orders if mentioned\n- Follow-Up if mentioned\n\nSeparate each problem with one blank line.\n\n# STYLE RULES\n\n- Use concise telegraphic bullets only\n- No full sentences unless necessary for clarity\n- No commentary, explanation, or preamble\n- Output ONLY the note\n- Do not use markdown formatting — no asterisks, no pound signs, plain text only\n- Include only information explicitly stated or clearly implied\n- Do not invent diagnoses, medications, labs, imaging, or follow-up\n- Preserve clinician wording when reasonable\n- Keep diagnoses in order mentioned\n- Do not create empty categories or placeholder bullets\n- Medication names and doses must match dictation exactly — omit dose if not stated\n- Use the explicit diagnosis or condition name as the heading, not presenting symptoms\n- If the clinician states a diagnosis, always prefer it over symptom descriptors as the heading\n\n# FORMATTING RULES\n\n- Differentials format:\n  Differential includes X, Y, Z\n\n- Return precautions format:\n  Return precautions include...\n\n- Follow-up format:\n  Follow-Up: ...\n\n# BOILERPLATE TAGS\n\nAfter all problem blocks, emit the appropriate tag(s) on their own line when the condition is present.\nDo not write the boilerplate text yourself — emit only the tag exactly as shown.\n\n{BOILERPLATE_TRIGGER_LIST}\n\nMultiple tags may apply. Each tag goes on its own line after the last problem block.\n\n# EXAMPLES\n\nDictation: "patient has acute otitis media, plan to treat with amoxicillin 90mg per kg per day divided twice daily, also tylenol motrin and hydration, return precautions for worsening fever or pain, follow up as needed"\n\nAcute Otitis Media\n- Amoxicillin 90mg/kg/day divided BID\n- Tylenol, Motrin, hydration\n- Return precautions include worsening fever, pain, failure to improve\n- Follow-Up: PRN\n[BOILERPLATE:ILLNESS]\n[BOILERPLATE:OTITIS]\n\n---\n\nDictation: "patient presenting with cough and fever, exam with right lower lobe crackles, diagnosis is community acquired pneumonia, treating with amoxicillin, also supportive care with tylenol motrin and fluids, return precautions for increased work of breathing, follow up as needed"\n\nCommunity-Acquired Pneumonia, right lower lobe\n- Amoxicillin\n- Tylenol, Motrin, fluids\n- Return precautions include increased work of breathing\n- Follow-Up: PRN\n[BOILERPLATE:ILLNESS]\n[BOILERPLATE:RESP]\n\n---\n\nDictation: "ADHD combined type, increasing concerta from 18 to 27mg daily, placing counseling referral, follow up in three months"\n\nADHD, combined\n- Concerta increased from 18mg to 27mg PO daily\n- Counseling referral placed\n- Follow-Up: 3 months\n[BOILERPLATE:PCMH]\n\n---\n\nDictation: "well child check, growing and developing well, anticipatory guidance discussed, all questions addressed, follow up in one year"\n\nWell Child Check\n- Growing and developing well\n- Anticipatory guidance discussed\n- Questions addressed\n- Follow-Up: 1 year/PRN\n[BOILERPLATE:WCC]`,
  boilerplate: [
    { key: "WCC", trigger: "Well child check or health maintenance discussed", text: "All forms, labs, immunizations, and patient concerns reviewed and addressed appropriately. Screening questions, past medical history, past social history, medications, and growth chart reviewed. Age-appropriate anticipatory guidance reviewed and printed in AVS. Parent questions addressed." },
    { key: "ILLNESS", trigger: "Any illness (infection, virus, fever, etc.) discussed", text: "Recommended supportive care with OTC medications as needed. Return precautions given including increasing pain, worsening fever, dehydration, new symptoms, prolonged symptoms, worsening symptoms, and other concerns. Caregiver expressed understanding and agreement with treatment plan." },
    { key: "INJURY", trigger: "Any injury discussed", text: "Recommended supportive care with Tylenol, Motrin, rest, ice, compression, elevation, and gradual return to activity as appropriate. Return precautions given including increasing pain, swelling, or failure to improve." },
    { key: "OTITIS", trigger: "Ear infection (otitis media) discussed", text: "Risk of untreated otitis media includes persistent pain and fever, hearing loss, and mastoiditis." },
    { key: "STREP", trigger: "Strep throat or rapid strep test discussed", text: "Risk of untreated strep throat includes rheumatic fever and peritonsillar abscess. This problem is moderate risk due to pending lab results which may necessitate further pharmacologic management." },
    { key: "DEHYDRATION", trigger: "Dehydration, vomiting, diarrhea, or decreased urination discussed", text: "Patient is at risk for dehydration, which would warrant emergency room care or admission for IV fluids." },
    { key: "RESP", trigger: "Trouble breathing, wheezing, or respiratory distress discussed", text: "Patient is at risk for worsening respiratory distress and clinical deterioration, which would need emergency room care or hospital admission." },
    { key: "PCMH", trigger: "ADHD, weight concern, obesity, or strep throat discussed", text: "PCMH Reminder" }
  ]
};

// ─── Settings persistence ─────────────────────────────────────────────────
const STORAGE_KEY = "present_settings_v1";
function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULTS);
    const saved = JSON.parse(raw);
    return {
      llmModel:      saved.llmModel      ?? DEFAULTS.llmModel,
      whisperModel:  saved.whisperModel  ?? DEFAULTS.whisperModel,
      ephemeralMode: saved.ephemeralMode ?? DEFAULTS.ephemeralMode,
      deidentifyInput: saved.deidentifyInput ?? DEFAULTS.deidentifyInput,
      autoCopyOutput: saved.autoCopyOutput ?? DEFAULTS.autoCopyOutput,
      // support both old key (medicationVocabulary) and new key (termVocabulary) gracefully
      termVocabulary: Array.isArray(saved.termVocabulary)
        ? saved.termVocabulary
        : Array.isArray(saved.medicationVocabulary)
          ? saved.medicationVocabulary
          : structuredClone(DEFAULTS.termVocabulary),
      cleanupPrompt: saved.cleanupPrompt ?? DEFAULTS.cleanupPrompt,
      mainPrompt:    saved.mainPrompt    ?? DEFAULTS.mainPrompt,
      boilerplate:   Array.isArray(saved.boilerplate) ? saved.boilerplate : structuredClone(DEFAULTS.boilerplate)
    };
  } catch { return structuredClone(DEFAULTS); }
}
function saveSettingsToStorage(s) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); return true; }
  catch { return false; }
}
let settings = loadSettings();

// ─── Settings accessors ───────────────────────────────────────────────────
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
    .map(e => `- ${e.trigger.trim()} \u2192 [BOILERPLATE:${e.key.trim().toUpperCase()}]`)
    .join("\n");
}
function getCleanupSystemPrompt() {
  let base = settings.cleanupPrompt;
  const vocab = (settings.termVocabulary || []).map(v => v.trim()).filter(Boolean);
  if (vocab.length > 0) {
    const list = vocab.map(t => `- ${t}`).join("\n");
    base += `\n\n# MEDICAL TERMINOLOGY ANCHORING\nThe following terms are commonly used in this clinical setting. If you encounter any word that appears phonetically garbled, oddly spelled, or out of place in a medical context, check whether it could plausibly be one of the terms below and correct it if confident. If uncertain, preserve the original word rather than guessing.\n\n${list}`;
  }
  return base;
}
function getNoteSystemPrompt() {
  return settings.mainPrompt.replace("{BOILERPLATE_TRIGGER_LIST}", buildBoilerplateTriggerList());
}

// ─── State ────────────────────────────────────────────────────────────────
let engine = null, transcriber = null;
let isLLMReady = false, isWhisperReady = false, isModelReady = false;
let isRecording = false, mediaRecorder = null, audioChunks = [];
let transcript = "", timerInterval = null, timerSeconds = 0, currentTab = "mic";
window._rawOutput = "";

function setPrivacyStatus(text = "") {
  const badge = document.getElementById("privacyStatus");
  if (!badge) return;
  const parts = [];
  parts.push(settings.ephemeralMode ? "Ephemeral" : "Settings saved");
  if (settings.deidentifyInput) parts.push("De-ID");
  if (!settings.autoCopyOutput) parts.push("Clipboard manual");
  badge.textContent = text || parts.join(" / ");
  badge.title = settings.ephemeralMode
    ? "Clinical text is kept in page memory only and cleared when you clear, reload, or close the tab."
    : "Settings persist in this browser. Clinical input/output is still not written by the app.";
}

function notifyServiceWorker(message) {
  if (!navigator.serviceWorker?.controller) return;
  navigator.serviceWorker.controller.postMessage(message);
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  try {
    const registration = await navigator.serviceWorker.register("./service-worker.js");
    if (registration.active && !navigator.serviceWorker.controller) {
      setPrivacyStatus("Offline shell ready after reload");
    }
  } catch (err) {
    console.warn("Service worker registration failed:", err);
  }
}

function activateNetworkLockdown() {
  notifyServiceWorker({ type: "SET_NETWORK_LOCKDOWN", enabled: true });
  setPrivacyStatus("Models ready / network locked");
}

function redactIdentifiers(text) {
  return text
    .replace(/\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b/g, "[SSN]")
    .replace(/\b(?:MRN|medical record number)\s*[:#-]?\s*[A-Z0-9-]+\b/gi, "[MRN]")
    .replace(/\b(?:DOB|date of birth)\s*[:#-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b/gi, "[DOB]")
    .replace(/\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b/g, "[DATE]")
    .replace(/\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g, "[PHONE]")
    .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[EMAIL]")
    .replace(/\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5}\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl)\b\.?/gi, "[ADDRESS]");
}

function clearSensitiveMemory() {
  transcript = "";
  audioChunks = [];
  window._rawOutput = "";
}

function checkModelsReady() {
  if (isLLMReady && isWhisperReady) {
    isModelReady = true;
    setStatus("ready", "Models ready");
    showProgress(false);
    activateNetworkLockdown();
    updateProcessBtn();
  }
}

// ─── Model Init ───────────────────────────────────────────────────────────
async function initModel() {
  isLLMReady = false; isWhisperReady = false; isModelReady = false;
  engine = null; transcriber = null;
  const whisperMeta = WHISPER_MODELS[settings.whisperModel] || WHISPER_MODELS[DEFAULTS.whisperModel];
  const llmMeta     = LLM_MODELS[settings.llmModel]         || LLM_MODELS[DEFAULTS.llmModel];
  setStatus("loading", "Initializing models\u2026");
  showProgress(true, `Downloading models (LLM ${llmMeta.size}, Whisper ${whisperMeta.size})\u2026`, 0);

  pipeline("automatic-speech-recognition", settings.whisperModel, {
    device: whisperMeta.device,
    progress_callback: (p) => {
      if (p.status === "progress" && !isLLMReady)
        showProgress(true, `Loading ${whisperMeta.label}\u2026`, Math.round(p.progress || 0));
    }
  }).then(t => { transcriber = t; isWhisperReady = true; checkModelsReady(); })
    .catch(err => { console.error("Whisper init failed:", err); showError("Failed to load Whisper model. Try a smaller model in Settings."); });

  try {
    engine = await webllm.CreateMLCEngine(settings.llmModel, {
      initProgressCallback: (p) => {
        showProgress(true, p.text || `Loading ${llmMeta.label}\u2026`, Math.round((p.progress || 0) * 100));
      }
    });
    isLLMReady = true; checkModelsReady();
  } catch (err) {
    console.error("Model init failed:", err);
    setStatus("error", "Model failed to load");
    showProgress(false);
    showError("Failed to load the AI model. Try a smaller model in Settings, or check that your browser supports WebGPU (Chrome 113+ recommended).");
  }
}

// ─── UI Helpers ───────────────────────────────────────────────────────────
function setStatus(state, text) {
  document.getElementById("statusDot").className = "status-dot " + state;
  document.getElementById("statusText").textContent = text;
}
function showProgress(visible, label = "", pct = 0) {
  const wrap = document.getElementById("progressWrap");
  if (visible) {
    wrap.style.display = "block";
    document.getElementById("progressBar").style.width = pct + "%";
    document.getElementById("progressLabel").textContent = label;
  } else { wrap.style.display = "none"; }
}
function showError(msg) {
  document.getElementById("outputEmpty").style.display = "none";
  document.getElementById("outputContent").style.display = "none";
  document.getElementById("outputStreaming").style.display = "none";
  document.getElementById("outputArea").innerHTML = `<div class="error-msg">${msg}</div>`;
}
function updateProcessBtn() {
  const hasContent = currentTab === "mic"
    ? (document.getElementById("transcriptText").value || transcript).trim().length > 20
    : document.getElementById("textInput").value.trim().length > 20;
  document.getElementById("btnProcess").disabled = !isModelReady || !hasContent;
}
window.switchTab = function(tab) {
  currentTab = tab;
  document.getElementById("micTab").classList.toggle("hidden", tab !== "mic");
  document.getElementById("textTab").classList.toggle("hidden", tab !== "text");
  document.getElementById("tabMic").classList.toggle("active", tab === "mic");
  document.getElementById("tabText").classList.toggle("active", tab === "text");
  updateProcessBtn();
};

// ─── Audio Recording ──────────────────────────────────────────────────────
async function setupAudioRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      audioChunks = [];
      await transcribeAudio(blob);
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
  ta.value = ""; ta.placeholder = "Transcribing securely in browser\u2026";
  ta.readOnly = true; ta.classList.remove("is-editable");
  document.getElementById("transcriptLabel").textContent = "Transcript";
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const result = await transcriber(audioBuffer.getChannelData(0));
    transcript = result.text.trim();
    if (transcript) {
      ta.value = transcript; ta.readOnly = false; ta.classList.add("is-editable");
      document.getElementById("transcriptLabel").textContent = "Transcript \u2014 editable";
      ta.oninput = () => { transcript = ta.value; updateProcessBtn(); };
    } else { ta.value = ""; ta.placeholder = "No speech detected."; }
    document.getElementById("micHint").textContent = "Click to begin recording";
    updateProcessBtn();
  } catch (err) {
    console.error("Transcription error:", err);
    ta.value = ""; ta.placeholder = "Failed to transcribe audio.";
    ta.readOnly = true; ta.classList.remove("is-editable");
    document.getElementById("micHint").textContent = "Click to begin recording";
  }
}
window.toggleRecording = async function() { if (isRecording) stopRecording(); else await startRecording(); };
async function startRecording() {
  if (!mediaRecorder) { await setupAudioRecording(); if (!mediaRecorder) return; }
  audioChunks = []; mediaRecorder.start(); isRecording = true;
  document.getElementById("micVisualizer").classList.add("recording");
  document.getElementById("micBtn").style.cssText = "";
  document.getElementById("micHint").textContent = "Listening\u2026 speak your assessment and plan";
  document.getElementById("recordingTimer").style.display = "flex";
  const ta = document.getElementById("transcriptText");
  ta.value = ""; ta.placeholder = "Recording in progress\u2026 Processing starts when you stop.";
  ta.readOnly = true; ta.classList.remove("is-editable");
  document.getElementById("transcriptLabel").textContent = "Transcript";
  timerSeconds = 0; updateTimer();
  timerInterval = setInterval(updateTimer, 1000);
}
function stopRecording() {
  isRecording = false; mediaRecorder.stop();
  document.getElementById("micVisualizer").classList.remove("recording");
  document.getElementById("micHint").textContent = "Processing audio locally...";
  document.getElementById("recordingTimer").style.display = "none";
  clearInterval(timerInterval);
}
function updateTimer() {
  timerSeconds++;
  document.getElementById("timerDisplay").textContent =
    `${String(Math.floor(timerSeconds/60)).padStart(2,"0")}:${String(timerSeconds%60).padStart(2,"0")}`;
}

// ─── Post-processing ──────────────────────────────────────────────────────
function stripThinkTags(raw) {
  return raw.replace(/<think>[\s\S]*?<\/think>/gi, "").replace(/<think>[\s\S]*/gi, "").trim();
}
function applyBoilerplate(raw) {
  const bpMap = getBoilerplateMap();
  return raw.replace(/\[BOILERPLATE:([A-Z_]+)\]/g, (_, key) => bpMap[key] ? "\n" + bpMap[key] + "\n" : "");
}
function postProcess(raw) { return applyBoilerplate(stripThinkTags(raw)); }

// ─── Toast / Clipboard ────────────────────────────────────────────────────
function autoCopyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => showAutocopyToast("\u2713 Auto-copied to clipboard")).catch(() => {});
}
function showAutocopyToast(msg) {
  const toast = document.getElementById("autocopyToast");
  toast.textContent = msg; toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 2500);
}
function getCurrentOutputText() {
  const lines = [];
  for (const node of document.getElementById("outputContent").children) {
    if (node.classList.contains("problem-block")) {
      const t = node.querySelector(".problem-title"); if (t) lines.push(t.textContent.trim());
      node.querySelectorAll(".problem-items li").forEach(li => lines.push("- " + li.textContent.trim()));
      lines.push("");
    } else if (node.classList.contains("boilerplate-block")) {
      lines.push(node.textContent.trim()); lines.push("");
    } else { const t = node.textContent.trim(); if (t) lines.push(t); }
  }
  return lines.join("\n").trim();
}

// ─── Process Note ─────────────────────────────────────────────────────────
window.processNote = async function() {
  if (!isModelReady || !engine) return;
  let input = currentTab === "mic"
    ? document.getElementById("transcriptText").value.trim()
    : document.getElementById("textInput").value.trim();
  if (!input) return;
  if (settings.deidentifyInput) input = redactIdentifiers(input);
  const empty = document.getElementById("outputEmpty");
  const content = document.getElementById("outputContent");
  const streaming = document.getElementById("outputStreaming");
  const streamText = document.getElementById("streamText");
  const btnCopy = document.getElementById("btnCopy");
  const btnCopyGroup = document.getElementById("btnCopyGroup");
  const editHint = document.getElementById("editHint");
  const thinkLabel = document.querySelector(".thinking-label");
  empty.style.display = "none"; content.style.display = "none"; content.contentEditable = "false";
  streaming.style.display = "block"; btnCopy.style.display = "none";
  if (btnCopyGroup) btnCopyGroup.style.display = "none";
  editHint.style.display = "none"; streamText.textContent = "";
  document.getElementById("btnProcess").disabled = true;
  try {
    thinkLabel.innerHTML = '<span class="think-dot"></span>Cleaning transcript\u2026';
    const cleanupResponse = await engine.chat.completions.create({
      messages: [
        { role: "system", content: getCleanupSystemPrompt() },
        { role: "user", content: input + "\n\n/no_think" }
      ],
      stream: false, temperature: 0.1, max_tokens: 512, extra_body: { enable_thinking: false }
    });
    let cleanedInput = stripThinkTags(cleanupResponse.choices[0]?.message?.content || input);
    if (!cleanedInput.trim()) cleanedInput = input;
    if (currentTab === "mic") {
      transcript = cleanedInput;
      const ta = document.getElementById("transcriptText");
      ta.value = cleanedInput; ta.readOnly = false; ta.classList.add("is-editable");
      document.getElementById("transcriptLabel").textContent = "Transcript \u2014 editable";
      ta.oninput = () => { transcript = ta.value; updateProcessBtn(); };
    } else { document.getElementById("textInput").value = cleanedInput; }
    thinkLabel.innerHTML = '<span class="think-dot"></span>Generating notes\u2026';
    streamText.textContent = "";
    let rawText = "";
    const stream = await engine.chat.completions.create({
      messages: [
        { role: "system", content: getNoteSystemPrompt() },
        { role: "user", content: `Convert this clinical dictation into structured assessment and plan notes:\n\n${cleanedInput}\n\n/no_think` }
      ],
      stream: true, temperature: 0.1, max_tokens: 1024, extra_body: { enable_thinking: false }
    });
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || "";
      rawText += delta; streamText.textContent = rawText;
    }
    const processedText = postProcess(rawText);
    streaming.style.display = "none"; renderOutput(processedText);
    if (btnCopyGroup) btnCopyGroup.style.display = "flex";
    btnCopy.style.display = "flex"; editHint.style.display = "flex";
    window._rawOutput = processedText;
    setTimeout(() => { content.contentEditable = "true"; }, 200);
    if (settings.autoCopyOutput) {
      setTimeout(() => { autoCopyToClipboard(processedText); }, 400);
    }
  } catch (err) {
    console.error("Generation error:", err);
    streaming.style.display = "none";
    showError("Error generating notes: " + err.message);
  }
  updateProcessBtn();
};

// ─── Render Output ────────────────────────────────────────────────────────
function isBoilerplateParagraph(text) { return text.length > 60 && !text.startsWith("-") && !text.startsWith("\u2022"); }
function renderOutput(raw) {
  const content = document.getElementById("outputContent");
  content.style.display = "block"; content.innerHTML = "";
  let currentItems = null, blockCount = 0, problemCount = 0;
  for (const line of raw.split("\n")) {
    const trimmed = line.trim(); if (!trimmed) continue;
    const isBullet = trimmed.startsWith("-") || trimmed.startsWith("\u2022");
    if (isBoilerplateParagraph(trimmed)) {
      const bp = document.createElement("div");
      bp.className = "boilerplate-block"; bp.style.opacity = "0"; bp.textContent = trimmed;
      content.appendChild(bp); currentItems = null; blockCount++;
      requestAnimationFrame(() => { bp.style.opacity = "1"; });
    } else if (!isBullet) {
      problemCount++;
      const block = document.createElement("div");
      block.className = "problem-block"; block.dataset.problemIndex = problemCount;
      block.style.animationDelay = `${blockCount * 0.08}s`; block.style.opacity = "0";
      const titleRow = document.createElement("div"); titleRow.className = "problem-title-row";
      const title = document.createElement("div"); title.className = "problem-title"; title.textContent = trimmed;
      titleRow.appendChild(title);
      const copyBtn = document.createElement("button");
      copyBtn.className = "btn-copy-problem"; copyBtn.setAttribute("aria-label","Copy this problem block");
      copyBtn.title = "Copy this problem"; copyBtn.dataset.problemIndex = problemCount;
      copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
      copyBtn.addEventListener("click", (e) => { e.stopPropagation(); copyProblemBlock(block, copyBtn); });
      titleRow.appendChild(copyBtn); block.appendChild(titleRow);
      const ul = document.createElement("ul"); ul.className = "problem-items";
      block.appendChild(ul); content.appendChild(block); currentItems = ul; blockCount++;
      requestAnimationFrame(() => { block.style.opacity = ""; });
    } else if (isBullet && currentItems) {
      const li = document.createElement("li"); li.textContent = trimmed.replace(/^[-\u2022]\s*/, "");
      currentItems.appendChild(li);
    }
  }
  if (blockCount === 0) {
    content.innerHTML = `<pre style="font-family:var(--font-mono);font-size:0.8rem;line-height:1.8;color:var(--text);padding:1rem;white-space:pre-wrap">${raw}</pre>`;
  }
  updateCopyDropdown();
}

// ─── Copy ─────────────────────────────────────────────────────────────────
function flashCopied(btn, resetHTML) {
  btn.classList.add("copied");
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><polyline points="20 6 9 17 4 12"/></svg>`;
  setTimeout(() => { btn.classList.remove("copied"); btn.innerHTML = resetHTML; }, 1800);
}
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
function getProblemBlockText(block) {
  const lines = [];
  const title = block.querySelector(".problem-title"); if (title) lines.push(title.textContent.trim());
  block.querySelectorAll(".problem-items li").forEach(li => lines.push("- " + li.textContent.trim()));
  return lines.join("\n");
}
function copyProblemBlock(block, btn) {
  const resetHTML = btn.innerHTML;
  navigator.clipboard.writeText(getProblemBlockText(block))
    .then(() => { flashCopied(btn, resetHTML); showAutocopyToast("\u2713 Problem copied"); })
    .catch(() => {});
}
window.copyProblemByIndex = function(idx) {
  const block = document.querySelector(`.problem-block[data-problem-index="${idx}"]`);
  if (!block) return;
  navigator.clipboard.writeText(getProblemBlockText(block))
    .then(() => { showAutocopyToast("\u2713 Problem " + idx + " copied"); closeCopyDropdown(); })
    .catch(() => {});
};
function updateCopyDropdown() {
  const dropdown = document.getElementById("copyDropdown"); if (!dropdown) return;
  dropdown.innerHTML = "";
  const allOpt = document.createElement("button");
  allOpt.className = "copy-dropdown-item copy-dropdown-all"; allOpt.setAttribute("role","option");
  allOpt.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy All`;
  allOpt.addEventListener("click", () => { copyOutput(); closeCopyDropdown(); });
  dropdown.appendChild(allOpt);
  const blocks = document.querySelectorAll(".problem-block");
  if (blocks.length > 1) {
    const divider = document.createElement("div"); divider.className = "copy-dropdown-divider";
    dropdown.appendChild(divider);
    blocks.forEach((block, i) => {
      const titleEl = block.querySelector(".problem-title");
      const label = titleEl ? titleEl.textContent.trim() : `Problem ${i+1}`;
      const opt = document.createElement("button");
      opt.className = "copy-dropdown-item"; opt.setAttribute("role","option"); opt.dataset.problemIndex = i+1;
      opt.innerHTML = `<span class="copy-dropdown-num">${i+1}</span>${label.length>34?label.slice(0,32)+"\u2026":label}`;
      opt.addEventListener("click", () => window.copyProblemByIndex(i+1));
      dropdown.appendChild(opt);
    });
  }
}
window.toggleCopyDropdown = function(e) {
  e.stopPropagation();
  const dropdown = document.getElementById("copyDropdown");
  const chevron = document.getElementById("btnCopyChevron");
  if (dropdown.classList.contains("open")) closeCopyDropdown();
  else { dropdown.classList.add("open"); chevron.setAttribute("aria-expanded","true"); }
};
function closeCopyDropdown() {
  const d = document.getElementById("copyDropdown"), c = document.getElementById("btnCopyChevron");
  if (d) d.classList.remove("open"); if (c) c.setAttribute("aria-expanded","false");
}
document.addEventListener("click", () => closeCopyDropdown());

// ─── Clear ────────────────────────────────────────────────────────────────
window.clearAll = function() {
  clearSensitiveMemory();
  const ta = document.getElementById("transcriptText");
  ta.value = ""; ta.placeholder = "Your words will appear here as you speak\u2026";
  ta.readOnly = true; ta.classList.remove("is-editable"); ta.oninput = null;
  document.getElementById("transcriptLabel").textContent = "Transcript";
  document.getElementById("textInput").value = "";
  document.getElementById("outputEmpty").style.display = "flex";
  const content = document.getElementById("outputContent");
  content.style.display = "none"; content.contentEditable = "false"; content.innerHTML = "";
  document.getElementById("outputStreaming").style.display = "none";
  document.getElementById("btnCopy").style.display = "none";
  const cg = document.getElementById("btnCopyGroup"); if (cg) cg.style.display = "none";
  document.getElementById("editHint").style.display = "none";
  const area = document.getElementById("outputArea");
  const err = area.querySelector(".error-msg"); if (err) err.remove();
  area.appendChild(document.getElementById("outputEmpty"));
  area.appendChild(content);
  area.appendChild(document.getElementById("outputStreaming"));
  window._rawOutput = "";
  if (isRecording) {
    const prev = mediaRecorder.onstop; mediaRecorder.onstop = null;
    stopRecording(); mediaRecorder.onstop = prev;
  }
  updateProcessBtn();
};
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
  document.getElementById("settingLLMModel").value = settings.llmModel;
  document.getElementById("settingWhisperModel").value = settings.whisperModel;
  document.getElementById("settingEphemeralMode").checked = !!settings.ephemeralMode;
  document.getElementById("settingDeidentifyInput").checked = !!settings.deidentifyInput;
  document.getElementById("settingAutoCopyOutput").checked = !!settings.autoCopyOutput;
  document.getElementById("settingTermVocabulary").value = (settings.termVocabulary || []).join("\n");
  document.getElementById("settingCleanupPrompt").value = settings.cleanupPrompt;
  document.getElementById("settingMainPrompt").value = settings.mainPrompt;
  renderBoilerplateList();
}

// ─── Reload Models ────────────────────────────────────────────────────────
window.reloadModels = function() {
  settings.llmModel     = document.getElementById("settingLLMModel").value;
  settings.whisperModel = document.getElementById("settingWhisperModel").value;
  saveSettingsToStorage(settings);
  const note = document.getElementById("settingsModelNote");
  if (note) note.textContent = "Reloading models\u2026";
  closeSettings();
  clearAll();
  initModel().finally(() => { if (note) note.textContent = ""; });
};

// ─── Boilerplate UI ───────────────────────────────────────────────────────
function renderBoilerplateList() {
  const container = document.getElementById("boilerplateList");
  container.innerHTML = "";
  settings.boilerplate.forEach((entry, idx) => container.appendChild(createBoilerplateEntryEl(entry, idx)));
}
function escHtml(str) {
  return String(str||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function createBoilerplateEntryEl(entry, idx) {
  const div = document.createElement("div"); div.className = "bp-entry"; div.dataset.idx = idx;
  div.innerHTML = `
    <div class="bp-entry-header" onclick="toggleBpEntry(this)">
      <span class="bp-entry-key-badge">[BOILERPLATE:<input class="bp-entry-key-input" type="text" value="${escHtml(entry.key)}" placeholder="KEY" maxlength="30" spellcheck="false" onclick="event.stopPropagation()" oninput="updateBpKey(${idx},this.value)"/>]</span>
      <span class="bp-entry-toggle"><svg class="bp-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></span>
    </div>
    <div class="bp-entry-body">
      <div><div class="bp-field-label">Trigger phrase(s)</div><textarea class="bp-entry-trigger" rows="2" placeholder="e.g. Well child check or health maintenance discussed" oninput="updateBpField(${idx},'trigger',this.value)">${escHtml(entry.trigger)}</textarea></div>
      <div><div class="bp-field-label">Boilerplate text</div><textarea class="bp-entry-text" rows="4" oninput="updateBpField(${idx},'text',this.value)">${escHtml(entry.text)}</textarea></div>
      <div class="bp-entry-footer"><button class="bp-btn-delete" onclick="deleteBpEntry(${idx})"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>Delete</button></div>
    </div>`;
  return div;
}
window.toggleBpEntry = (h) => h.closest(".bp-entry").classList.toggle("expanded");
window.updateBpKey   = (i,v) => { settings.boilerplate[i].key = v.toUpperCase().replace(/[^A-Z0-9_]/g,""); };
window.updateBpField = (i,f,v) => { settings.boilerplate[i][f] = v; };
window.deleteBpEntry = (i) => { settings.boilerplate.splice(i,1); renderBoilerplateList(); };
window.addBoilerplateEntry = function() {
  settings.boilerplate.push({key:"",trigger:"",text:""});
  renderBoilerplateList();
  const last = document.getElementById("boilerplateList").lastElementChild;
  if (last) { last.classList.add("expanded"); last.querySelector(".bp-entry-key-input")?.focus(); }
};
window.saveSettings = function() {
  settings.llmModel      = document.getElementById("settingLLMModel").value;
  settings.whisperModel  = document.getElementById("settingWhisperModel").value;
  settings.ephemeralMode = document.getElementById("settingEphemeralMode").checked;
  settings.deidentifyInput = document.getElementById("settingDeidentifyInput").checked;
  settings.autoCopyOutput = document.getElementById("settingAutoCopyOutput").checked;
  settings.termVocabulary = document.getElementById("settingTermVocabulary").value
    .split("\n").map(s => s.trim()).filter(Boolean);
  settings.cleanupPrompt = document.getElementById("settingCleanupPrompt").value;
  settings.mainPrompt    = document.getElementById("settingMainPrompt").value;
  const ok = saveSettingsToStorage(settings);
  setPrivacyStatus();
  const status = document.getElementById("settingsSaveStatus");
  status.textContent = ok ? "\u2713 Saved \u2014 click Reload Models to apply model changes" : "\u26a0 Could not persist (storage blocked)";
  status.classList.add("visible");
  setTimeout(() => status.classList.remove("visible"), 3500);
};
window.resetSettings = function() {
  if (!confirm("Reset all settings to defaults? This cannot be undone.")) return;
  settings = structuredClone(DEFAULTS);
  saveSettingsToStorage(settings);
  setPrivacyStatus();
  populateSettingsUI();
  const status = document.getElementById("settingsSaveStatus");
  status.textContent = "\u2713 Reset to defaults \u2014 click Reload Models to apply";
  status.classList.add("visible");
  setTimeout(() => status.classList.remove("visible"), 3500);
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && document.getElementById("settingsDrawer").classList.contains("open")) closeSettings();
});

// ─── Boot ─────────────────────────────────────────────────────────────────
setPrivacyStatus();
registerServiceWorker();
window.addEventListener("beforeunload", () => {
  notifyServiceWorker({ type: "SET_NETWORK_LOCKDOWN", enabled: false });
  if (settings.ephemeralMode) clearSensitiveMemory();
});
initModel();
