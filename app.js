import * as webllm from "./vendor/web-llm/web-llm.esm.js";
import { pipeline, env } from "./vendor/transformers/transformers.min.js";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "whisper-models/";
env.useBrowserCache = false;

const LLM_MODELS = {
  "gemma-2-2b-it-q4f32_1-MLC":         { label: "Gemma 2 2B", size: "~1.3 GB" },
  "Phi-3.5-mini-instruct-q4f16_1-MLC": { label: "Phi 3.5 Mini", size: "~2.2 GB" },
  "Qwen3-4B-q4f16_1-MLC":              { label: "Qwen3 4B", size: "~2.3 GB" },
};

const WHISPER_MODELS = {
  "Xenova/whisper-tiny.en":  { label: "Whisper Tiny", size: "~75 MB", device: "wasm" },
  "Xenova/whisper-small.en": { label: "Whisper Small", size: "~488 MB", device: "wasm" },
};

// Unified Library (replaces isolated boilerplate and templates)
const DEFAULT_TEMPLATES = [
  {
    id: "WELL_CHILD", name: "Well Child / Health Maintenance", type: "text", priority: 1,
    triggers: ["well child", "wcc", "checkup", "physical"],
    content: "<em>All forms, labs, immunizations, and patient concerns reviewed and addressed appropriately. Screening questions, past medical history, past social history, medications, and growth chart reviewed. Age-appropriate anticipatory guidance reviewed and printed in AVS. Parent questions addressed.</em>"
  },
  {
    id: "ILLNESS", name: "Illness Supportive Care", type: "text", priority: 2,
    triggers: ["illness", "sick", "fever", "cough", "congestion", "uri", "rash", "vomiting"],
    content: "<em>Recommended supportive care with OTC medications as needed. Return precautions given including increasing pain, worsening fever, dehydration, new symptoms, prolonged symptoms, worsening symptoms, and other concerns. Caregiver expressed understanding and agreement with treatment plan.</em>"
  },
  {
    id: "OTITIS", name: "Ear Infection Risk", type: "text", priority: 3,
    triggers: ["ear infection", "otitis", "ear pain"],
    content: "<em>Risk of untreated otitis media includes persistent pain and fever, hearing loss, and mastoiditis.</em>"
  },
  {
    id: "FOLLOW_UP", name: "Follow-Up Dropdown", type: "dropdown", priority: 10,
    triggers: ["follow up", "followup"], label: "Follow-Up", join: "lines", singleSelect: true,
    options: ["Follow up as needed.", "Follow up in 2-3 days.", "Follow up in 2-4 weeks.", "Follow up in 3 months.", "Follow up in 1 year."]
  }
];

const DEFAULTS = {
  llmModel: "Qwen3-4B-q4f16_1-MLC",
  whisperModel: "Xenova/whisper-small.en",
  termVocabulary: ["amoxicillin", "rocephin", "Tylenol", "Motrin"],
  macros: [{ key: ".aom", value: "acute otitis media" }, { key: ".rtp", value: "return precautions" }],
  templates: JSON.parse(JSON.stringify(DEFAULT_TEMPLATES)),
  smartNoteTemplate: "{input}\n\n{templates}",
  cleanupPrompt: "You are a medical transcription editor. Clean up the dictation without changing clinical meaning. Output only the transcript.",
  mainPrompt: `You are a clinical documentation assistant generating telegraphic assessment and plan notes.
- Use concise bullets.
- Output ONLY the note.
- Do not invent info.

# BOILERPLATE TAGS
After problem blocks, emit the appropriate tag(s) when the condition is present.
{BOILERPLATE_TRIGGER_LIST}
`
};

const STORAGE_KEY = "present_unified_settings";
let settings = loadSettings();
let currentTab = "mic";

let engine = null, transcriber = null;
let isLLMReady = false, isWhisperReady = false, isModelReady = false;
let isRecording = false, mediaRecorder = null, audioChunks = [];
let transcript = "";

// State for SmartChart engine
let scState = {
  activeDropdowns: [],
  dropdownSelections: {},
  currentNoteHtml: "",
  previewDebounce: null
};

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULTS);
    const saved = JSON.parse(raw);
    return {
      llmModel: saved.llmModel ?? DEFAULTS.llmModel,
      whisperModel: saved.whisperModel ?? DEFAULTS.whisperModel,
      termVocabulary: saved.termVocabulary ?? DEFAULTS.termVocabulary,
      macros: saved.macros ?? DEFAULTS.macros,
      templates: Array.isArray(saved.templates) ? saved.templates : structuredClone(DEFAULTS.templates),
      smartNoteTemplate: saved.smartNoteTemplate ?? DEFAULTS.smartNoteTemplate,
      cleanupPrompt: saved.cleanupPrompt ?? DEFAULTS.cleanupPrompt,
      mainPrompt: saved.mainPrompt ?? DEFAULTS.mainPrompt
    };
  } catch { return structuredClone(DEFAULTS); }
}

function saveSettingsToStorage(s) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); return true; } catch { return false; }
}

// ─── Helpers: Smart Templates & Post-Processing ────────────────────────────

function sanitizeHtml(html) {
  if (typeof DOMPurify !== "undefined") return DOMPurify.sanitize(String(html||""));
  return String(html||"").replace(/<[^>]*>/g, "");
}

function escapeRegExp(str) { return String(str).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function buildBoilerplateTriggerList() {
  return settings.templates
    .filter(t => t.type !== "dropdown") // Let LLM only trigger static text tags
    .map(t => `- ${(t.triggers || []).join(' or ')} -> [TEMPLATE:${t.id}]`)
    .join("\n");
}

function getNoteSystemPrompt() {
  return settings.mainPrompt.replace("{BOILERPLATE_TRIGGER_LIST}", buildBoilerplateTriggerList());
}

function postProcessLLM(raw) {
  let cleaned = raw.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  // Replace [TEMPLATE:ID] with actual template content
  cleaned = cleaned.replace(/\[TEMPLATE:([a-zA-Z0-9_-]+)\]/g, (_, id) => {
    const t = settings.templates.find(x => x.id === id);
    return t ? `\n<div class="boilerplate-block">${t.content}</div>\n` : "";
  });
  return cleaned;
}

function getDropdownSelection(id) {
  if (!scState.dropdownSelections[id]) scState.dropdownSelections[id] = { values: [], join: null };
  return scState.dropdownSelections[id];
}

function joinTemplateOptions(options, mode) {
  if (mode === 'lines') return '<ul>' + options.map(o => `<li>${o}</li>`).join('') + '</ul>';
  if (mode === 'sentence') return options.join(' ');
  return options.join(', ');
}

function renderDropdownValueHtml(t) {
  const selection = getDropdownSelection(t.id);
  const selected = Array.isArray(selection.values) ? selection.values : [];
  if (selected.length === 0) return '';
  return `<p><strong>${t.label || t.name}:</strong></p>${joinTemplateOptions(selected, selection.join || t.join || 'lines')}`;
}

function updateSmartPreview() {
  const input = document.getElementById("smartInput").value;
  const statusText = document.getElementById("sc-status-text");
  
  if (!input.trim()) {
    document.getElementById("sc-preview-rendered").innerHTML = "";
    statusText.textContent = "Ready";
    statusText.classList.remove("sc-matched");
    scState.activeDropdowns = [];
    scState.currentNoteHtml = "";
    return;
  }

  // Find matches
  const lowerInput = input.toLowerCase();
  const matched = settings.templates.filter(t => 
    (t.triggers || []).some(trig => {
      const escaped = escapeRegExp(trig.trim().toLowerCase()).replace(/\s+/g, '\\s+');
      return new RegExp(`(^|[^a-z0-9])${escaped}($|[^a-z0-9])`, 'i').test(lowerInput);
    })
  ).sort((a, b) => (a.priority ?? 99) - (b.priority ?? 99));

  let modifiedInput = input;
  const bottomTemplates = [];

  matched.forEach(t => {
    let replaced = false;
    if (t.type === 'dropdown') {
      t.triggers.forEach(trig => {
        const regex = new RegExp(`(^|[^a-z0-9])(${escapeRegExp(trig.trim())})($|[^a-z0-9])`, 'gi');
        if (regex.test(modifiedInput)) {
          modifiedInput = modifiedInput.replace(regex, `$1{dropdown:${t.id}}$3`);
          replaced = true;
        }
      });
    }
    if (!replaced) bottomTemplates.push(t);
  });

  // Extract active dropdowns
  scState.activeDropdowns = matched.filter(t => t.type === 'dropdown');

  // Build Output
  const templatesHtml = bottomTemplates.map(t => t.content).filter(Boolean).join('<hr class="sc-template-sep">');
  const escapedInput = modifiedInput.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
  
  let outHtml = settings.smartNoteTemplate
    .replace(/\{input\}/g, escapedInput)
    .replace(/\{templates\}/g, templatesHtml)
    .replace(/\{dropdown:([a-zA-Z0-9_-]+)\}/g, (_, id) => {
      const t = settings.templates.find(x => x.id === id);
      return t ? renderDropdownValueHtml(t) : '';
    });

  scState.currentNoteHtml = sanitizeHtml(outHtml);
  document.getElementById("sc-preview-rendered").innerHTML = scState.currentNoteHtml;
  
  // Render Dropdown Controls
  renderSmartDropdownControls();

  statusText.textContent = matched.length > 0 ? `${matched.length} template(s) matched` : "No matches found";
  statusText.classList.toggle("sc-matched", matched.length > 0);
  
  // Auto-copy for SmartChart (1.5s debounce)
  clearTimeout(scState.autoCopyTimer);
  scState.autoCopyTimer = setTimeout(() => {
    if (currentTab === 'smart' && scState.currentNoteHtml) copyOutput(scState.currentNoteHtml);
  }, 1500);
}

function renderSmartDropdownControls() {
  const container = document.getElementById("sc-preview-rendered");
  if (!container || scState.activeDropdowns.length === 0) return;

  const wrap = document.createElement('div');
  wrap.className = 'sc-dropdown-template-list';

  scState.activeDropdowns.forEach(t => {
    const selection = getDropdownSelection(t.id);
    const card = document.createElement('div');
    card.className = 'sc-dropdown-template';
    
    card.innerHTML = `<div class="sc-dropdown-template-header">${t.label || t.name}</div>`;
    const controls = document.createElement('div');
    controls.className = 'sc-dropdown-template-controls';

    const select = document.createElement('select');
    select.className = 'sc-dropdown-template-select';
    
    if (t.singleSelect) {
      select.innerHTML = `<option value="">— select one —</option>` + (t.options||[]).map(o => `<option value="${o}" ${selection.values.includes(o)?'selected':''}>${o}</option>`).join('');
    } else {
      select.multiple = true;
      select.size = Math.min((t.options||[]).length, 4);
      select.innerHTML = (t.options||[]).map(o => `<option value="${o}" ${selection.values.includes(o)?'selected':''}>${o}</option>`).join('');
    }

    const join = document.createElement('select');
    join.className = 'sc-dropdown-template-join';
    join.style.display = t.singleSelect ? 'none' : 'block';
    join.innerHTML = `<option value="lines">Bullets</option><option value="comma">Comma</option><option value="sentence">Sentence</option>`;
    join.value = selection.join || t.join || 'lines';

    const onChange = () => {
      selection.values = t.singleSelect ? (select.value ? [select.value] : []) : Array.from(select.selectedOptions).map(o=>o.value);
      selection.join = join.value;
      updateSmartPreview(); // Re-render
    };
    
    select.addEventListener('change', onChange);
    join.addEventListener('change', onChange);
    
    controls.append(select, join);
    card.append(controls);
    wrap.appendChild(card);
  });
  container.appendChild(wrap);
}

// ─── UI & Tabs ─────────────────────────────────────────────────────────────

window.switchTab = function(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.id === `tab${tab.charAt(0).toUpperCase() + tab.slice(1)}`));
  document.getElementById("micTab").classList.toggle("hidden", tab !== "mic");
  document.getElementById("textTab").classList.toggle("hidden", tab !== "text");
  document.getElementById("smartTab").classList.toggle("hidden", tab !== "smart");
  
  // Toggle output views
  const isSmart = tab === "smart";
  document.getElementById("llmView").style.display = isSmart ? "none" : "block";
  document.getElementById("smartView").style.display = isSmart ? "block" : "none";
  document.getElementById("outputEmpty").style.display = "none";
  
  const btnProcess = document.getElementById("btnProcess");
  if (isSmart) {
    btnProcess.style.display = "none";
    document.getElementById("btnCopyGroup").style.display = "flex";
  } else {
    btnProcess.style.display = "block";
    updateProcessBtn();
  }
};

function updateProcessBtn() {
  if (currentTab === 'smart') return;
  const hasContent = currentTab === "mic" ? document.getElementById("transcriptText").value.trim().length > 10 : document.getElementById("textInput").value.trim().length > 10;
  document.getElementById("btnProcess").disabled = !isModelReady || !hasContent;
}

// ─── Transcription & LLM (Mic / Text Tabs) ──────────────────────────────────

function setStatus(state, text) {
  document.getElementById("statusDot").className = "status-dot " + state;
  document.getElementById("statusText").textContent = text;
}
function showProgress(visible, label = "", pct = 0) {
  const wrap = document.getElementById("progressWrap");
  if (visible) { wrap.style.display = "block"; document.getElementById("progressBar").style.width = pct + "%"; document.getElementById("progressLabel").textContent = label; }
  else { wrap.style.display = "none"; }
}

async function initModel() {
  isLLMReady = false; isWhisperReady = false; isModelReady = false;
  setStatus("loading", "Initializing models…");
  showProgress(true, `Downloading models…`, 0);

  pipeline("automatic-speech-recognition", settings.whisperModel, {
    device: WHISPER_MODELS[settings.whisperModel]?.device || "wasm",
    local_files_only: true,
    progress_callback: (p) => { if (p.status === "progress" && !isLLMReady) showProgress(true, `Loading Whisper…`, Math.round(p.progress || 0)); }
  }).then(t => { transcriber = t; isWhisperReady = true; checkReady(); }).catch(console.error);

  try {
    engine = await webllm.CreateMLCEngine(settings.llmModel, {
      appConfig: { model_list: [{ model_url: "models/" + settings.llmModel, model_id: settings.llmModel, model_lib_url: "models/" + settings.llmModel + "/webgpu.wasm" }] },
      initProgressCallback: (p) => showProgress(true, p.text || `Loading LLM…`, Math.round((p.progress || 0) * 100))
    });
    isLLMReady = true; checkReady();
  } catch (err) { console.error("Model init failed:", err); setStatus("error", "Model failed"); showProgress(false); }
}

function checkReady() {
  if (isLLMReady && isWhisperReady) {
    isModelReady = true; setStatus("ready", "Models ready"); showProgress(false); updateProcessBtn();
  }
}

window.processNote = async function() {
  if (!isModelReady || !engine || currentTab === 'smart') return;
  const rawInput = currentTab === "mic" ? document.getElementById("transcriptText").value : document.getElementById("textInput").value;
  if (!rawInput.trim()) return;

  document.getElementById("btnProcess").disabled = true;
  document.getElementById("outputContent").style.display = "none";
  document.getElementById("outputStreaming").style.display = "block";
  document.getElementById("btnCopyGroup").style.display = "none";
  
  try {
    let cleanedInput = rawInput;
    if (settings.cleanupPrompt) {
      document.querySelector(".thinking-label").textContent = "Cleaning transcript...";
      const cleanRes = await engine.chat.completions.create({
        messages: [{ role: "system", content: settings.cleanupPrompt }, { role: "user", content: rawInput }],
        temperature: 0.1, max_tokens: 512
      });
      cleanedInput = cleanRes.choices[0]?.message?.content || rawInput;
    }

    document.querySelector(".thinking-label").textContent = "Generating notes...";
    let rawText = "";
    const stream = await engine.chat.completions.create({
      messages: [{ role: "system", content: getNoteSystemPrompt() }, { role: "user", content: cleanedInput }],
      stream: true, temperature: 0.1, max_tokens: 1024
    });

    for await (const chunk of stream) {
      rawText += chunk.choices[0]?.delta?.content || "";
      document.getElementById("streamText").textContent = rawText;
    }

    const processedHtml = postProcessLLM(rawText);
    document.getElementById("outputStreaming").style.display = "none";
    
    const contentBox = document.getElementById("outputContent");
    contentBox.style.display = "block";
    contentBox.innerHTML = processedHtml.replace(/\n/g, '<br>');
    contentBox.contentEditable = "true";
    document.getElementById("btnCopyGroup").style.display = "flex";
    
  } catch (err) { console.error(err); }
  updateProcessBtn();
};

window.copyOutput = function(overrideHtml = null) {
  const html = overrideHtml || document.getElementById(currentTab === 'smart' ? 'sc-preview-rendered' : 'outputContent').innerHTML;
  if (!html.trim()) return;
  
  const blobHtml = new Blob([html], { type: "text/html" });
  const blobText = new Blob([sanitizeHtml(html)], { type: "text/plain" });
  
  navigator.clipboard.write([new ClipboardItem({ "text/html": blobHtml, "text/plain": blobText })]).then(() => {
    const toast = document.getElementById("autocopyToast");
    toast.textContent = "✓ Copied to clipboard";
    toast.classList.add("visible");
    setTimeout(() => toast.classList.remove("visible"), 2000);
  });
};

window.clearAll = function() {
  document.getElementById("transcriptText").value = "";
  document.getElementById("textInput").value = "";
  document.getElementById("smartInput").value = "";
  document.getElementById("outputContent").innerHTML = "";
  document.getElementById("sc-preview-rendered").innerHTML = "";
  document.getElementById("btnCopyGroup").style.display = "none";
  document.getElementById("outputEmpty").style.display = "flex";
  updateProcessBtn();
  updateSmartPreview();
};

// ─── Settings UI (Template Builder) ────────────────────────────────────────

window.openSettings = function() {
  document.getElementById("settingLLMModel").value = settings.llmModel;
  document.getElementById("settingWhisperModel").value = settings.whisperModel;
  document.getElementById("settingTermVocabulary").value = settings.termVocabulary.join("\n");
  document.getElementById("settingMacros").value = settings.macros.map(m => `${m.key}: ${m.value}`).join("\n");
  document.getElementById("settingCleanupPrompt").value = settings.cleanupPrompt;
  document.getElementById("settingMainPrompt").value = settings.mainPrompt;
  document.getElementById("settingSmartNoteTemplate").value = settings.smartNoteTemplate;
  
  renderTemplateList();
  document.getElementById("settingsDrawer").classList.add("open");
  document.getElementById("settingsOverlay").classList.add("open");
};

window.closeSettings = () => {
  document.getElementById("settingsDrawer").classList.remove("open");
  document.getElementById("settingsOverlay").classList.remove("open");
};

function renderTemplateList() {
  const list = document.getElementById("sc-template-list");
  list.innerHTML = settings.templates.map((t, i) => `
    <div class="sc-template-item">
      <div>
        <div class="sc-template-name">${t.name} ${t.type==='dropdown'?'(Dropdown)':''}</div>
        <div class="sc-template-triggers">${(t.triggers||[]).join(', ')}</div>
      </div>
      <div>
        <button class="sc-btn-icon" onclick="editTemplate(${i})">Edit</button>
        <button class="sc-btn-icon" onclick="deleteTemplate(${i})">Del</button>
      </div>
    </div>
  `).join("");
}

window.openTemplateForm = function(idx = -1) {
  const form = document.getElementById("sc-template-form");
  document.getElementById("sc-template-list").classList.add("hidden");
  document.getElementById("sc-add-template-btn").classList.add("hidden");
  form.classList.remove("hidden");
  
  if (idx >= 0) {
    const t = settings.templates[idx];
    document.getElementById("sc-form-id").value = t.id;
    document.getElementById("sc-form-name").value = t.name;
    document.getElementById("sc-form-triggers").value = (t.triggers||[]).join(", ");
    document.getElementById("sc-form-type").value = t.type || "text";
    document.getElementById("sc-form-content").value = t.content || "";
    document.getElementById("sc-form-dropdown-label").value = t.label || "";
    document.getElementById("sc-form-options").value = (t.options||[]).join("\n");
    document.getElementById("sc-form-join").value = t.join || "lines";
    document.getElementById("sc-form-single-select").checked = !!t.singleSelect;
    document.getElementById("sc-form-category").value = t.category || "";
  } else {
    document.getElementById("sc-form-id").value = "tpl_" + Date.now();
    form.querySelectorAll("input[type=text], textarea").forEach(el => el.value = "");
  }
  updateTemplateTypeFields();
};

window.closeTemplateForm = function() {
  document.getElementById("sc-template-form").classList.add("hidden");
  document.getElementById("sc-template-list").classList.remove("hidden");
  document.getElementById("sc-add-template-btn").classList.remove("hidden");
};

window.updateTemplateTypeFields = function() {
  const isDrop = document.getElementById("sc-form-type").value === "dropdown";
  document.getElementById("sc-form-text-fields").classList.toggle("hidden", isDrop);
  document.getElementById("sc-form-dropdown-fields").classList.toggle("hidden", !isDrop);
};

window.saveTemplateForm = function() {
  const id = document.getElementById("sc-form-id").value;
  const name = document.getElementById("sc-form-name").value.trim();
  const triggers = document.getElementById("sc-form-triggers").value.split(",").map(s => s.trim()).filter(Boolean);
  const type = document.getElementById("sc-form-type").value;
  
  if (!name || triggers.length === 0) return alert("Name and triggers are required.");

  const tpl = { id, name, triggers, type, priority: 10, category: document.getElementById("sc-form-category").value.trim() };
  
  if (type === "text") {
    tpl.content = document.getElementById("sc-form-content").value.trim();
  } else {
    tpl.label = document.getElementById("sc-form-dropdown-label").value.trim();
    tpl.options = document.getElementById("sc-form-options").value.split("\n").map(s=>s.trim()).filter(Boolean);
    tpl.join = document.getElementById("sc-form-join").value;
    tpl.singleSelect = document.getElementById("sc-form-single-select").checked;
  }

  const idx = settings.templates.findIndex(x => x.id === id);
  if (idx >= 0) settings.templates[idx] = tpl; else settings.templates.push(tpl);
  
  renderTemplateList();
  closeTemplateForm();
};

window.editTemplate = (i) => openTemplateForm(i);
window.deleteTemplate = (i) => { if(confirm("Delete this template?")) { settings.templates.splice(i,1); renderTemplateList(); } };

window.saveSettings = function() {
  settings.llmModel = document.getElementById("settingLLMModel").value;
  settings.whisperModel = document.getElementById("settingWhisperModel").value;
  settings.cleanupPrompt = document.getElementById("settingCleanupPrompt").value;
  settings.mainPrompt = document.getElementById("settingMainPrompt").value;
  settings.smartNoteTemplate = document.getElementById("settingSmartNoteTemplate").value;
  
  saveSettingsToStorage(settings);
  const status = document.getElementById("settingsSaveStatus");
  status.textContent = "✓ Saved";
  status.classList.add("visible");
  setTimeout(() => status.classList.remove("visible"), 2000);
  updateSmartPreview();
};

window.reloadModels = function() { saveSettings(); closeSettings(); initModel(); };

// Initialization
document.getElementById("smartInput").addEventListener("input", () => {
  clearTimeout(scState.previewDebounce);
  scState.previewDebounce = setTimeout(updateSmartPreview, 250);
});

switchTab('mic');
initModel();