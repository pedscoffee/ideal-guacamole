// ClinicalScribe — app.js
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
const MODEL_ID = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";

async function initModel() {
  setStatus("loading", "Initializing model…");
  showProgress(true, "Downloading model weights (first load ~700MB, cached after)…", 0);

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
      // Auto-restart for continuous listening
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

// ─── Process Note ─────────────────────────────────────────────────────────
window.processNote = async function() {
  if (!isModelReady || !engine) return;

  const input = currentTab === "mic"
    ? transcript.trim()
    : document.getElementById("textInput").value.trim();

  if (!input) return;

  // Show streaming UI
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

  const systemPrompt = `You are a clinical documentation assistant that converts clinician dictation into concise, telegraphic assessment and plan notes for each problem.

OUTPUT FORMAT — for each problem/diagnosis mentioned:
Diagnosis or Problem Name
- Brief telegraphic bullet about management
- Labs if ordered
- Imaging if ordered]
- Medications with doses if mentioned
- Supportive care
- Situational awareness / conditional orders if mentioned
- Return precautions include if mentioned
- Nursing orders if mentioned
- Follow-up timing if mentioned

RULES:
- Use telegraphic style: short, direct phrases, no full sentences, no unnecessary words
- Do not label problems with diagnosis or problem, just list the diagnosis or problem
- Only include bullet points for information actually mentioned in the dictation
- Do NOT add bullets for categories not mentioned
- List diagnoses in the order they appear in the dictation
- Medication names and doses should be exact as dictated
- If a differential is mentioned, format as: "Differential includes X, Y, and Z"
- If follow-up is mentioned, format as: Follow-Up:
- Return precautions start with: "Return precautions include..."
- Separate each problem with a blank line
- Do NOT include any explanation, preamble, or commentary — output ONLY the structured note

## Conditional Boilerplate Text

[Insert after all problem blocks and before the follow-up line when applicable. Add a blank line before and after each boilerplate statement.]

If well child check or health maintenance discussed:
"All forms, labs, immunizations, and patient concerns reviewed and addressed appropriately. Screening questions, past medical history, past social history, medications, and growth chart reviewed. Age-appropriate anticipatory guidance reviewed and printed in AVS. Parent questions addressed."

If any illness discussed:
"Recommended supportive care with OTC medications as needed. Return precautions given including increasing pain, worsening fever, dehydration, new symptoms, prolonged symptoms, worsening symptoms, and other concerns. Caregiver expressed understanding and agreement with treatment plan."

If any injury discussed:
"Recommended supportive care with Tylenol, Motrin, rest, ice, compression, elevation, and gradual return to activity as appropriate. Return precautions given including increasing pain, swelling, or failure to improve."

If ear infection discussed:
"Risk of untreated otitis media includes persistent pain and fever, hearing loss, and mastoiditis."

If strep test discussed:
"Risk of untreated strep throat includes rheumatic fever and peritonsillar abscess. This problem is moderate risk due to pending lab results which may necessitate further pharmacologic management."

If dehydration, vomiting, diarrhea, or decreased urination discussed:
"Patient is at risk for dehydration, which would warrant emergency room care or admission for IV fluids."

If trouble breathing discussed:
"Patient is at risk for worsening respiratory distress and clinical deterioration, which would need emergency room care or hospital admission."

If ADHD, weight, obesity, or strep throat discussed:
"PCMH Reminder"

## Few-Shot Examples

Acute Otitis Media
- Amoxicillin
- Tylenol, Motrin, and emphasis on hydration
- Return precautions include worsening fever, pain, or failure to improve
- Follow-Up: PRN

ADHD, combined
- Concerta 18mg increased to Concerta 27mg PO daily
- Counseling referral placed today
- Follow-Up: 3 months

Fever
- Differential includes Kawasaki disease, MIS-C, RMSF
- CBC, CMP, ESR, CRP, and UA
- Chest XR
- 1L NS bolus, IVIG
- Tylenol, Motrin, Zofran
- If decrease in BP, then will give another 1L NS bolus and consider ICU
- Vitals q4hr, call if change in rash

Well Child Check
- Growing and developing well
- Anticipatory guidance discussed
- All questions addressed
- Follow up: 1 year/PRN

Abnormal Well Child Check
- Growing well
- Speech delay noted, will refer for speech therapy and audiology
- Anticipatory guidance discussed
- All questions addressed
- Follow up: 1 year/PRN

Rash
- Differential includes ringworm, pityriasis rosea, and scabies
- Ketoconazole
- Zyrtec, atarax for itching and sleep
- Return precautions include worsening rash, worsening itch, and failure to improve
- If spreads further or fails to improve with ketoconazole may consider permethrin
- Follow-Up: PRN
`;

  const userPrompt = `Convert this clinical dictation into structured assessment and plan notes:\n\n${input}`;

  try {
    let fullText = "";
    const stream = await engine.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      stream: true,
      temperature: 0.1,
      max_tokens: 1024
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || "";
      fullText += delta;
      streamText.textContent = fullText;
    }

    // Parse and render structured output
    streaming.style.display = "none";
    renderOutput(fullText);
    btnCopy.style.display = "flex";
    window._rawOutput = fullText;

  } catch (err) {
    console.error("Generation error:", err);
    streaming.style.display = "none";
    showError("Error generating notes: " + err.message);
  }

  updateProcessBtn();
};

// ─── Render Output ────────────────────────────────────────────────────────
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
    if (!isBullet && trimmed.length > 0) {
      // New problem heading
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

      // Trigger animation
      requestAnimationFrame(() => {
        block.style.opacity = "";
      });
    } else if (isBullet && currentItems) {
      const li = document.createElement("li");
      li.textContent = trimmed.replace(/^[-•]\s*/, "");
      currentItems.appendChild(li);
    } else if (!currentBlock) {
      // Fallback: create a generic block
      const block = document.createElement("div");
      block.className = "problem-block";
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
    }
  }

  // If nothing was parsed, show raw
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
  // Remove any error divs
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
