// Present — app.js
// WebLLM + Web Speech API for clinical A&P note generation

import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// ─── State ────────────────────────────────────────────────────────────────
let engine = null;
let isModelReady = false;
let isRecording = false;
let recognition = null;
let transcript = "";
let timerInterval = null;
let timerSeconds = 0;
let currentTab = "mic";

// ─── Model Setup ──────────────────────────────────────────────────────────
const MODEL_ID = "Qwen3-4B-q4f16_1-MLC";

async function initModel() {
  setStatus("loading", "Initializing model…");
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

// ─── Speech Recognition ───────────────────────────────────────────────────
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    document.getElementById("micHint").textContent = "Speech recognition not supported in this browser. Please use the Text tab.";
    document.getElementById("micBtn").disabled = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  let finalTranscript = "";

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (result.isFinal) {
        finalTranscript += result[0].transcript + " ";
      } else {
        interim += result[0].transcript;
      }
    }
    transcript = finalTranscript;
    const display = finalTranscript + (interim ? `<em style="color:var(--text-dim)">${interim}</em>` : "");
    document.getElementById("transcriptText").innerHTML = display || '<span class="placeholder">Your words will appear here as you speak…</span>';
    updateProcessBtn();
  };

  recognition.onerror = (e) => {
    if (e.error !== "aborted") {
      console.warn("Speech error:", e.error);
      stopRecording();
    }
  };

  recognition.onend = () => {
    if (isRecording) {
      try { recognition.start(); } catch (_) {}
    }
  };
}

window.toggleRecording = function() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
};

function startRecording() {
  if (!recognition) {
    setupSpeechRecognition();
    if (!recognition) return;
  }
  isRecording = true;
  document.getElementById("micVisualizer").classList.add("recording");
  document.getElementById("micBtn").style.cssText = "";
  document.getElementById("micHint").textContent = "Listening… speak your assessment and plan";
  document.getElementById("recordingTimer").style.display = "flex";
  timerSeconds = 0;
  updateTimer();
  timerInterval = setInterval(updateTimer, 1000);
  try { recognition.start(); } catch (_) {}
}

function stopRecording() {
  isRecording = false;
  document.getElementById("micVisualizer").classList.remove("recording");
  document.getElementById("micHint").textContent = "Click to begin recording";
  document.getElementById("recordingTimer").style.display = "none";
  clearInterval(timerInterval);
  try { recognition.stop(); } catch (_) {}
  updateProcessBtn();
}

function updateTimer() {
  timerSeconds++;
  const m = String(Math.floor(timerSeconds / 60)).padStart(2, "0");
  const s = String(timerSeconds % 60).padStart(2, "0");
  document.getElementById("timerDisplay").textContent = `${m}:${s}`;
}

// ─── Post-Processors ──────────────────────────────────────────────────────

// 1. Strip <think>...</think> blocks that Qwen3 may emit despite /no_think.
//    Handles multiline blocks and malformed unclosed tags defensively.
function stripThinkTags(raw) {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*/gi, "")   // unclosed tag: drop everything after it
    .trim();
}

// 2. Replace [BOILERPLATE:KEY] tags with guaranteed-correct boilerplate text.
//    To update boilerplate, edit ONLY the BOILERPLATE object below.
//    Unknown keys are silently removed so stray tags never reach the output.
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

// Master post-process pipeline — order matters: strip think tags first,
// then inject boilerplate so tags inside think blocks can't accidentally fire.
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

  // /no_think appended to user prompt — Qwen3's native in-prompt thinking toggle,
  // more reliable than extra_body in WebLLM context.
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
      // Show live stream; think tags visible during generation but stripped on completion
      streamText.textContent = rawText;
    }

    // Post-process: strip think tags, then inject boilerplate
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
// Boilerplate paragraphs are long prose sentences injected by postProcess().
// They get a distinct italic dim style vs. short diagnosis headings.
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
      // Render as distinct italic prose block, not a diagnosis heading
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
      // Diagnosis / problem heading
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
setupSpeechRecognition();
initModel();
