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

const PIPELINE_LIBRARY = [
  {
    id: "avs",
    title: "ETA: After Visit Summary (AVS) Generation",
    label: "AVS / Sign-Off",
    description: "Generates personalized sign-offs + actionable family to-do lists. Change few shot examples to match what you see and your voice.",
    specialty: "Pediatrics (Easy to change)",
    category: "Documentation",
    order: 5,
    inputSource: "noteOutput",
    prompt: `Sign-Off & Family To-Do Generator
Generate two components: 3 personalized sign-off options and a family to-do list.

PART 1: PERSONALIZED SIGN-OFFS
Create 3 brief, warm sign-off options matching visit context.
Structure: [Personal touch/acknowledgment], [well-wish or next step]. [Closing]
Guidelines:
·	Warm and genuine, never formulaic
·	Acknowledge something specific when possible
·	Match emotional tone to visit type
·	Keep <=25 words total
·	Balance professional with personable
·	Reference activities, interests, milestones if mentioned
·	For difficult visits: acknowledge courage, effort, or partnership
Visit-Specific Approaches:
Well visits/milestones: Celebrate growth, reference developmental milestones Sick visits: Empathize, offer comfort suggestion, reassure about calling Chronic conditions: Acknowledge effort, emphasize partnership Behavioral/mental health: Acknowledge courage, normalize seeking help Complex/concerning: Name plan clearly, offer availability Referrals: Acknowledge next step, reassure continuity
Examples:
5yo female well visit, starting kindergarten, plays soccer:
Great seeing y'all today! Good luck with soccer and kindergarten! Excited for you!
She is going to do great in kindergarten. Can't wait to hear about it next visit!
What an exciting year ahead! We're here if you need anything.
Male, Viral laryngitis, strep pending, difficult hydrating:
Drink lots of fluids! Popsicles are my favorite when sick. Call if not improving.
Hope he feels better soon. Those popsicles should help. We'll call with results tomorrow.
It is hard when they feel so crummy. Keep offering fluids and call if he's not gettting better. We're here!
Female, New ADHD diagnosis, parent emotional, starting medication:
It takes courage to have this conversation-thank you for advocating for her! We're here to support you.
You're doing the right thing getting her help. We'll partner with you every step. Call anytime.
A new diagnosis can feel overwhelming, but we'll take it step by step. We're here for you.
Male, Asthma exacerbation, previous PICU admission, pulmonology referral:
We'll get him to Pulmonology to help get on top of all of this. Call if anything changes-we're here.
It is scary navigating all of this after your experiences. The specialist will help us care for him. Please call with any concerns.
You're doing everything right seeking care early. Don't hesitate to reach out.
Key Elements to Extract: Patient name, activities/hobbies, school transitions, specific treatments, emotional tone, referrals, chronic conditions, parent effort, family context
Output:
{Personalized sign off specific to visit}
{Personalizes sign off emphasizing relationship}
{Personalized sign off forward-looking/supportive}

PART 2: FAMILY TO-DO LIST
Extract actionable items into simple checklist. Include only concrete next steps with essential details.
Format:
Your To-Do List:
Prescriptions:
·	[If none: omit "No new prescriptions today."]
Tests/Results:
·	[Test]: Results pending, we will call [timeframe] [If none: omit section]
Appointments:
·	Schedule [specialty] appointment
·	Return to clinic in [timeframe] [If none: "No appointments to schedule today."]

Rules:
1.	Simple dashes for bullets
2.	One item per line, <=12 words
3.	Only items requiring family action
4.	Be specific about timeframes
5.	Bold section headers
6.	No explanations, just actions
7.	Extract from note only
8.  Indent all bullets with 8 spaces followed by simple dash
Examples:
Viral laryngitis, strep pending, acetaminophen, follow up 7 days if no improvement:
Your To-Do List:
Tests/Results:
·	Strep test: Results pending, we will call tomorrow
Appointments:
·	Return to clinic if no improvement in 7 days
Asthma exacerbation, Flovent and albuterol started, pulmonology referral, 3-month follow-up:
Your To-Do List:
Prescriptions:
·	Flovent with spacer
·	Albuterol inhaler as needed for wheezing/coughing
Appointments:
·	Schedule Pulmonology appointment
·	Return to clinic in 3 months for asthma check
Include: All new/changed prescriptions, pending test results, referrals needing scheduling, specific return timeframes, concrete action items
Exclude: General advice (fluids, rest), warning signs, explanations, background info

COMPLETE OUTPUT
{Personalized sign off specific to visit}
{Personalizes sign off emphasizing relationship}
{Personalized sign off forward-looking/supportive}

To-Do List:
Prescriptions: [list or "No new prescriptions today."]
Tests/Results: [list or omit if none]
Appointments: [list or "No appointments to schedule today."]`
  },
  {
    id: "billing",
    title: "Billing Analysis",
    label: "Billing Analysis",
    description: "Assesses MDM components and suggests CPT E/M codes with detailed reasoning.",
    specialty: "Pediatrics (adaptable but may need some work)",
    category: "Administrative",
    order: 6,
    inputSource: "noteOutput",
    prompt: `Analyze this note and determine the appropriate CPT E/M billing code using 2021 E/M guidelines for an ESTABLISHED patient.
MDM Component Assessment
A. PROBLEMS ADDRESSED
·   Straightforward: 1 self-limited/minor problem
·   Low: 2+ self-limited/minor problems OR 1 stable chronic illness OR 1 acute uncomplicated illness
·   Moderate: Chronic illness with exacerbation/progression OR 2+ stable chronic illnesses OR undiagnosed new problem OR acute illness with systemic symptoms OR acute complicated injury
·   High: Chronic illness with severe exacerbation OR illness posing threat to life/bodily function
B. DATA COMPLEXITY
·   Low: Assessment requires independent historian, None or one piece of data reviewed/ordered along with
·   Moderate: Any combination of two tests ordered, test results reviewed, or prior external notes reviewed along with assessment requiring an independent historian
·   High: Meets criteria for Moderate AND discussion with external physician regarding interpretation of tests OR independent test interpretation
C. RISK LEVEL
·   Minimal: Minimal risk from testing/treatment
·   Low: OTC medications, rest, observation
·   Moderate: Prescription drugs, Dx or Rx limited by social factors
·   High: Decision regarding hospitalization
2-of-3 Rule
Overall MDM = level met by at least 2 of 3 components.
·   Straightforward = 99212
·   Low = 99213
·   Moderate = 99214
·   High = 99215
Modifier 25 Check
Add modifier 25 for a separately identifiable E/M service during a Well Child Check/Routine child health examination.
Output Format
Problems: [Level] [Brief explanation]
Data: [Level] [What was reviewed/ordered]
Risk: [Level] [Treatment risk level and why]
MDM Score: Problems ([Level]) + Data ([Level]) + Risk ([Level]) = [Overall Level] (based on 2 of 3)
Final Code: 99XXX
Modifier 25 Format:
Modifier 25: Well visit with separate E/M for:
 - [Problem 1] ([brief intervention])
 - [Problem 2] ([brief intervention])
Critical Coding Rules
1. Ordering any culture (e.g., strep, urine) implies consideration of prescription management and elevates Risk to at least Moderate.
2. Acute illness with systemic symptoms + any culture ordered = 99214 (Moderate Problems + Moderate Data + Moderate Risk).
3. Assume Assessment requiring an independent historian is always true.
Examples
Viral URI (simple) Runny nose, cough. Exam: clear. Plan: supportive care.
Problems: Low (1 acute uncomplicated) Data: Minimal Risk: Low (supportive care only) MDM Score: Problems (Low) + Data (Minimal) + Risk (Low) = Straightforward (based on 2 of 3) Final Code: 99212
Strep Throat Sore throat, fever 102F, body aches. Exam: exudates. Plan: strep test, amox if positive.
Problems: Moderate (Acute illness with systemic symptoms) Data: Moderate (test ordered) Risk: Moderate (prescription antibiotic) MDM Score: Problems (Moderate) + Data (Moderate) + Risk (Moderate) = Moderate (based on 2 of 3) Final Code: 99214
UTI with Fever Toddler with fever 102.5, crying with urination. Exam: suprapubic tenderness. Urine dipstick positive. Plan: send urine culture.
Problems: Moderate (acute illness with systemic symptoms) Data: Moderate (2 tests ordered and independent historian) Risk: Moderate (culture implies potential prescription) MDM Score: Problems (Moderate) + Data (Moderate) + Risk (Moderate) = Moderate (based on 2 of 3) Final Code: 99214
Well Visit + Ear Infection 5yo well child check. Parent reports ear pain, fever x2 days. Exam: acute otitis media. Plan: amoxicillin.
Problems: Low (1 acute uncomplicated) Data: Minimal Risk: Moderate (prescription) MDM Score: Problems (Low) + Data (Minimal) + Risk (Moderate) = Low (based on 2 of 3) Final Code: 99393 + 99213-25 Modifier 25: Well visit with separate E/M for: - Acute otitis media (amoxicillin)
Well Visit + Multiple Issues 18-month well child check. Also has URI and diaper rash. Exam: clear rhinorrhea, diaper dermatitis. Plan: supportive care for URI, barrier cream for rash.
Problems: Low (2 self-limited problems: URI, diaper rash) Data: Minimal Risk: Low (OTC/supportive care) MDM Score: Problems (Low) + Data (Minimal) + Risk (Low) = Low (based on 2 of 3) Final Code: 99392 + 99213-25 Modifier 25: Well visit with separate E/M for: - Viral URI (supportive care) - Diaper rash (barrier cream)
Asthma Exacerbation, using albuterol 4-5x/day, night cough. Exam: mild wheezing. Plan: increase Flovent.
Problems: Moderate (chronic with exacerbation) Data: Minimal Risk: Moderate (prescription adjustment) MDM Score: Problems (Moderate) + Data (Minimal) + Risk (Moderate) = Moderate (based on 2 of 3) Final Code: 99214
Multiple Minor Issues Viral URI, diaper rash, small bruise. Exam unremarkable. Plan: supportive care, barrier cream, observation.
Problems: Low (3 self-limited problems) Data: Minimal Risk: Low (OTC only) MDM Score: Problems (Low) + Data (Minimal) + Risk (Low) = Straightforward (based on 2 of 3) Final Code: 99212
*Do not list any references that were used*`
  },
  {
    id: "teaching",
    title: "Teaching - Socratic Prompt",
    label: "Teaching - Socratic",
    description: "Extracts clinical pearl and then asks a follow up question to probe a student's understanding further.",
    specialty: "All",
    category: "Teaching",
    order: 10,
    inputSource: "noteOutput",
    prompt: `From this case, extract one brief 'Clinical Pearl' for teaching (<=20 words). Focus on a practical pitfall, tip, or insight-not patient-specific.
Using that clinical pearl, then take it one step farther and ask the clinician a socratic style follow up question to cause them to think more deeply about this particular patient. Write each on a separate line below bolded header "Clinical Pearl:"`
  }
];

function defaultPipelineSteps() {
  return PIPELINE_LIBRARY.map(t => ({
    id: t.id,
    label: t.label,
    enabled: false,
    prompt: t.prompt,
    inputSource: t.inputSource
  }));
}

// ─── Default Settings ─────────────────────────────────────────────────────
const DEFAULTS = {
  llmModel: "Qwen3-4B-q4f16_1-MLC",
  whisperModel: "Xenova/whisper-small.en",
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
  ],
  macros: [
    { key: ".aom", value: "acute otitis media" },
    { key: ".sob", value: "shortness of breath" },
    { key: ".rtp", value: "return precautions" },
    { key: ".flu", value: "influenza" },
    { key: ".wcc", value: "well child check" },
    { key: ".pe", value: "physical exam" },
    { key: ".hpi", value: "history of present illness" }
  ],
  pipelineSteps: defaultPipelineSteps()
};

// ─── Settings persistence ─────────────────────────────────────────────────
const STORAGE_KEY = "present_settings_v2";
const LEGACY_STORAGE_KEY = "present_settings_v1";
function normalizePipelineSteps(steps) {
  if (!Array.isArray(steps)) return structuredClone(DEFAULTS.pipelineSteps);
  return steps.map((step, idx) => ({
    id: step.id || `custom-${Date.now()}-${idx}`,
    label: step.label || "Untitled Step",
    enabled: Boolean(step.enabled),
    prompt: step.prompt || "",
    inputSource: step.inputSource === "cleanedTranscript" ? "cleanedTranscript" : "noteOutput"
  }));
}
function loadSettings() {
  try {
    let raw = localStorage.getItem(STORAGE_KEY);
    const shouldMigrate = !raw;
    if (!raw) raw = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULTS);
    const saved = JSON.parse(raw);
    const hydrated = {
      llmModel:      saved.llmModel      ?? DEFAULTS.llmModel,
      whisperModel:  saved.whisperModel  ?? DEFAULTS.whisperModel,
      // support both old key (medicationVocabulary) and new key (termVocabulary) gracefully
      termVocabulary: Array.isArray(saved.termVocabulary)
        ? saved.termVocabulary
        : Array.isArray(saved.medicationVocabulary)
          ? saved.medicationVocabulary
          : structuredClone(DEFAULTS.termVocabulary),
      cleanupPrompt: saved.cleanupPrompt ?? DEFAULTS.cleanupPrompt,
      mainPrompt:    saved.mainPrompt    ?? DEFAULTS.mainPrompt,
      boilerplate:   Array.isArray(saved.boilerplate) ? saved.boilerplate : structuredClone(DEFAULTS.boilerplate),
      macros:        Array.isArray(saved.macros) ? saved.macros : structuredClone(DEFAULTS.macros),
      pipelineSteps: normalizePipelineSteps(saved.pipelineSteps)
    };
    if (shouldMigrate) saveSettingsToStorage(hydrated);
    return hydrated;
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

// ─── Shorthand Macros Engine ──────────────────────────────────────────────
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function expandMacros(text) {
  if (!text || !settings.macros || settings.macros.length === 0) return text;
  let result = text;
  const sortedMacros = [...settings.macros].sort((a, b) => b.key.length - a.key.length);
  for (const macro of sortedMacros) {
    if (!macro.key || !macro.value) continue;
    const escapedKey = escapeRegExp(macro.key);
    const regex = new RegExp('(^|\\s)' + escapedKey + '(?=\\s|[.,;:!?\\)]|$)', 'gi');
    result = result.replace(regex, `$1${macro.value}`);
  }
  return result;
}
function setupTextareaMacroExpander(textarea) {
  if (!textarea) return;
  textarea.addEventListener("keydown", (e) => {
    const triggers = [" ", "Enter", ".", ",", ";", ":", "?", "!", ")"];
    if (!triggers.includes(e.key)) return;
    const start = textarea.selectionStart;
    const textBefore = textarea.value.slice(0, start);
    const textAfter = textarea.value.slice(start);
    const words = textBefore.split(/(\s+)/);
    if (words.length === 0) return;
    const lastWord = words[words.length - 1].trim();
    if (!lastWord) return;
    const matchingMacro = (settings.macros || []).find(
      m => m.key.toLowerCase() === lastWord.toLowerCase()
    );
    if (matchingMacro) {
      e.preventDefault();
      const expandedValue = matchingMacro.value;
      const keyToInsert = e.key === "Enter" ? "\n" : e.key;
      words[words.length - 1] = expandedValue;
      const newTextBefore = words.join("") + keyToInsert;
      textarea.value = newTextBefore + textAfter;
      const newCursorPos = newTextBefore.length;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
      textarea.dispatchEvent(new Event("input"));
    }
  });
}
function initMacroExpanders() {
  setupTextareaMacroExpander(document.getElementById("textInput"));
  setupTextareaMacroExpander(document.getElementById("transcriptText"));
}


// ─── State ────────────────────────────────────────────────────────────────
let engine = null, transcriber = null;
let isLLMReady = false, isWhisperReady = false, isModelReady = false;
let isRecording = false, mediaRecorder = null, audioChunks = [], recordingStream = null;
let transcript = "", timerInterval = null, timerSeconds = 0, currentTab = "mic";

// Web Audio visualizer globals
let visualizerAudioCtx = null;
let visualizerAnalyser = null;
let visualizerSource = null;
let visualizerAnimationId = null;

function checkModelsReady() {
  if (isLLMReady && isWhisperReady) {
    isModelReady = true;
    setStatus("ready", "Models ready");
    showProgress(false);
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
  resetPipelineStepOutputs();
  const area = document.getElementById("outputArea");
  area.querySelector(".error-msg")?.remove();
  const error = document.createElement("div");
  error.className = "error-msg";
  error.textContent = msg;
  area.prepend(error);
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
    recordingStream = stream;
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
// ─── Circular Audio Waveform Visualizer ──────────────────────────────────────
function startVisualizer() {
  const stream = recordingStream || (mediaRecorder && mediaRecorder.stream);
  if (!stream) return;
  try {
    if (!visualizerAudioCtx) {
      visualizerAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } else if (visualizerAudioCtx.state === "suspended") {
      visualizerAudioCtx.resume();
    }
    visualizerAnalyser = visualizerAudioCtx.createAnalyser();
    visualizerAnalyser.fftSize = 256;
    visualizerSource = visualizerAudioCtx.createMediaStreamSource(stream);
    visualizerSource.connect(visualizerAnalyser);
    drawWaveform();
  } catch (err) {
    console.error("Failed to start waveform visualizer:", err);
  }
}

function drawWaveform() {
  const canvas = document.getElementById("micWaveform");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const centerX = width / 2;
  const centerY = height / 2;
  const baseRadius = 38; // Radius just outside the 56px button (which has 28px radius)
  const maxOffset = 25;  // Max outward animation amplitude

  const bufferLength = visualizerAnalyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  const smoothData = new Float32Array(bufferLength);

  function draw() {
    if (!isRecording) {
      ctx.clearRect(0, 0, width, height);
      return;
    }
    visualizerAnimationId = requestAnimationFrame(draw);
    visualizerAnalyser.getByteFrequencyData(dataArray);

    ctx.clearRect(0, 0, width, height);

    // Background ambient ring
    ctx.beginPath();
    ctx.arc(centerX, centerY, baseRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = "rgba(0, 229, 160, 0.08)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Draw glowing circular waveform
    ctx.beginPath();
    const numPoints = 64; // Balance resolution and rendering performance
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * 2 * Math.PI;
      // Symmetric styling: map left/right to keep circle visually balanced
      const dataIdx = i < numPoints / 2 ? i : numPoints - i;
      const rawValue = dataArray[dataIdx] || 0;

      // Smooth transitions
      smoothData[i] = smoothData[i] * 0.75 + rawValue * 0.25;
      const offset = (smoothData[i] / 255.0) * maxOffset;
      const r = baseRadius + offset;

      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();

    ctx.strokeStyle = "#00e5a0";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#00e5a0";
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0; // Reset shadow for other drawings
  }
  draw();
}

function stopVisualizer() {
  if (visualizerAnimationId) {
    cancelAnimationFrame(visualizerAnimationId);
    visualizerAnimationId = null;
  }
  if (visualizerSource) {
    visualizerSource.disconnect();
    visualizerSource = null;
  }
  if (visualizerAnalyser) {
    visualizerAnalyser = null;
  }
  if (visualizerAudioCtx && visualizerAudioCtx.state !== "closed") {
    visualizerAudioCtx.suspend();
  }
  const canvas = document.getElementById("micWaveform");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

window.toggleRecording = async function() { if (isRecording) stopRecording(); else await startRecording(); };
async function startRecording() {
  if (!mediaRecorder) { await setupAudioRecording(); if (!mediaRecorder) return; }
  audioChunks = []; mediaRecorder.start(); isRecording = true;
  startVisualizer();
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
  stopVisualizer();
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
function copyToClipboardRich(plainText, htmlText) {
  if (typeof ClipboardItem === "undefined") {
    return navigator.clipboard.writeText(plainText);
  }
  const plainTextBlob = new Blob([plainText], { type: "text/plain" });
  const htmlTextBlob = new Blob([htmlText], { type: "text/html" });
  const item = new ClipboardItem({
    "text/plain": plainTextBlob,
    "text/html": htmlTextBlob
  });
  return navigator.clipboard.write([item]);
}
function autoCopyToClipboard(plainText, htmlText) {
  copyToClipboardRich(plainText, htmlText || plainText)
    .then(() => showAutocopyToast("\u2713 Auto-copied to clipboard"))
    .catch(() => {});
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
function getCurrentOutputHtml() {
  const htmlParts = [];
  for (const node of document.getElementById("outputContent").children) {
    if (node.classList.contains("problem-block")) {
      const t = node.querySelector(".problem-title");
      if (t) {
        htmlParts.push(`<h3><strong>${escHtml(t.textContent.trim())}</strong></h3>`);
      }
      const lis = [];
      node.querySelectorAll(".problem-items li").forEach(li => {
        lis.push(`  <li>${escHtml(li.textContent.trim())}</li>`);
      });
      if (lis.length > 0) {
        htmlParts.push(`<ul>\n${lis.join("\n")}\n</ul>`);
      }
      htmlParts.push("<br/>");
    } else if (node.classList.contains("boilerplate-block")) {
      htmlParts.push(`<p><em>${escHtml(node.textContent.trim())}</em></p>`);
      htmlParts.push("<br/>");
    } else {
      const t = node.textContent.trim();
      if (t) {
        htmlParts.push(`<p>${escHtml(t)}</p>`);
      }
    }
  }
  return htmlParts.join("\n").trim();
}
function getProblemBlockHtml(block) {
  const htmlParts = [];
  const title = block.querySelector(".problem-title");
  if (title) {
    htmlParts.push(`<h3><strong>${escHtml(title.textContent.trim())}</strong></h3>`);
  }
  const lis = [];
  block.querySelectorAll(".problem-items li").forEach(li => {
    lis.push(`  <li>${escHtml(li.textContent.trim())}</li>`);
  });
  if (lis.length > 0) {
    htmlParts.push(`<ul>\n${lis.join("\n")}\n</ul>`);
  }
  return htmlParts.join("\n");
}
function getPipelineStepsContainer() {
  return document.getElementById("pipelineStepsOutput");
}
function resetPipelineStepOutputs() {
  const container = getPipelineStepsContainer();
  if (!container) return;
  container.innerHTML = "";
  container.style.display = "none";
}
function renderPipelineRichText(raw) {
  function inlineMarkdown(text) {
    return escHtml(text)
      .replace(/\*\*([^*\n][\s\S]*?[^*\n])\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
  }

  const html = [];
  let listType = null;
  const closeList = () => {
    if (listType) html.push(`</${listType}>`);
    listType = null;
  };

  for (const line of raw.split("\n")) {
    const bullet = line.match(/^\s*[-*•]\s+(.+)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (!line.trim()) {
      closeList();
      html.push("<div><br></div>");
    } else if (bullet) {
      if (listType !== "ul") {
        closeList();
        html.push("<ul>");
        listType = "ul";
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
    } else if (numbered) {
      if (listType !== "ol") {
        closeList();
        html.push("<ol>");
        listType = "ol";
      }
      html.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
    } else {
      closeList();
      html.push(`<div>${inlineMarkdown(line)}</div>`);
    }
  }
  closeList();
  return html.join("");
}
function renderPipelineStepOutput(step, text = "", state = "streaming") {
  const container = getPipelineStepsContainer();
  if (!container) return null;
  container.style.display = "flex";
  let box = container.querySelector(`[data-step-id="${CSS.escape(step.id)}"]`);
  if (!box) {
    box = document.createElement("section");
    box.className = "pipeline-step-output";
    box.dataset.stepId = step.id;
    box.innerHTML = `
      <div class="pipeline-step-header">
        <div class="pipeline-step-title-wrap">
          <span class="pipeline-step-kicker">EXTRA</span>
          <h3 class="pipeline-step-title"></h3>
        </div>
        <div class="pipeline-step-status"><span class="think-dot"></span>Generating...</div>
        <button class="btn-copy pipeline-step-copy" type="button">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy
        </button>
      </div>
      <div class="pipeline-step-text" contenteditable="false" spellcheck="true"></div>
    `;
    container.appendChild(box);
  }
  box.querySelector(".pipeline-step-title").textContent = step.label || "Pipeline Step";
  const textEl = box.querySelector(".pipeline-step-text");
  textEl.innerHTML = renderPipelineRichText(text);
  textEl.contentEditable = state === "done" ? "true" : "false";
  box.classList.toggle("is-streaming", state === "streaming");
  box.classList.toggle("is-error", state === "error");
  const status = box.querySelector(".pipeline-step-status");
  status.innerHTML = state === "streaming"
    ? '<span class="think-dot"></span>Generating...'
    : state === "error"
      ? "Could not generate"
      : "Ready";
  const copyBtn = box.querySelector(".pipeline-step-copy");
  copyBtn.disabled = !text.trim() || state === "streaming";
  copyBtn.onclick = () => copyPipelineStepOutput(step.id, copyBtn);
  return box;
}
async function copyPipelineStepOutput(stepId, btn) {
  const box = document.querySelector(`[data-step-id="${CSS.escape(stepId)}"]`);
  const textEl = box?.querySelector(".pipeline-step-text");
  const text = textEl?.innerText || textEl?.textContent || "";
  if (!text.trim()) return;
  const html = textEl?.innerHTML || escHtml(text);
  const resetHTML = btn.innerHTML;
  try {
    await copyToClipboardRich(text, `<div>${html}</div>`);
    flashCopied(btn, resetHTML);
    showAutocopyToast("\u2713 Step copied");
  } catch {}
}
async function runPipelineStep(step, userContent) {
  let rawText = "";
  renderPipelineStepOutput(step, "", "streaming");
  const stream = await engine.chat.completions.create({
    messages: [
      { role: "system", content: step.prompt },
      { role: "user", content: `${userContent}\n\n/no_think` }
    ],
    stream: true, temperature: 0.1, max_tokens: 1024, extra_body: { enable_thinking: false }
  });
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content || "";
    rawText += delta;
    renderPipelineStepOutput(step, stripThinkTags(rawText), "streaming");
  }
  const finalText = stripThinkTags(rawText);
  renderPipelineStepOutput(step, finalText, "done");
  return finalText;
}

// ─── Process Note ─────────────────────────────────────────────────────────
window.processNote = async function() {
  if (!isModelReady || !engine) return;
  const rawInput = currentTab === "mic"
    ? document.getElementById("transcriptText").value.trim()
    : document.getElementById("textInput").value.trim();
  if (!rawInput) return;
  const input = expandMacros(rawInput);
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
  resetPipelineStepOutputs();
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
    setTimeout(() => {
      const plainText = getCurrentOutputText() || processedText;
      const htmlText = getCurrentOutputHtml() || processedText;
      autoCopyToClipboard(plainText, htmlText);
    }, 400);
    const enabledSteps = (settings.pipelineSteps || []).filter(s => s.enabled && s.prompt.trim());
    for (const step of enabledSteps) {
      const userContent = step.inputSource === "cleanedTranscript" ? cleanedInput : processedText;
      try {
        await runPipelineStep(step, userContent);
      } catch (stepErr) {
        console.error("Pipeline step failed:", stepErr);
        renderPipelineStepOutput(step, "Error generating this pipeline step: " + stepErr.message, "error");
      }
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
  const plainText = getCurrentOutputText() || window._rawOutput || "";
  const htmlText = getCurrentOutputHtml() || window._rawOutput || "";
  copyToClipboardRich(plainText, htmlText).then(() => {
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
  const plainText = getProblemBlockText(block);
  const htmlText = getProblemBlockHtml(block);
  copyToClipboardRich(plainText, htmlText)
    .then(() => { flashCopied(btn, resetHTML); showAutocopyToast("\u2713 Problem copied"); })
    .catch(() => {});
}
window.copyProblemByIndex = function(idx) {
  const block = document.querySelector(`.problem-block[data-problem-index="${idx}"]`);
  if (!block) return;
  const plainText = getProblemBlockText(block);
  const htmlText = getProblemBlockHtml(block);
  copyToClipboardRich(plainText, htmlText)
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
  transcript = "";
  const ta = document.getElementById("transcriptText");
  ta.value = ""; ta.placeholder = "Your words will appear here as you speak\u2026";
  ta.readOnly = true; ta.classList.remove("is-editable"); ta.oninput = null;
  document.getElementById("transcriptLabel").textContent = "Transcript";
  document.getElementById("textInput").value = "";
  document.getElementById("outputEmpty").style.display = "flex";
  const content = document.getElementById("outputContent");
  content.style.display = "none"; content.contentEditable = "false"; content.innerHTML = "";
  document.getElementById("outputStreaming").style.display = "none";
  resetPipelineStepOutputs();
  document.getElementById("btnCopy").style.display = "none";
  const cg = document.getElementById("btnCopyGroup"); if (cg) cg.style.display = "none";
  document.getElementById("editHint").style.display = "none";
  const area = document.getElementById("outputArea");
  const err = area.querySelector(".error-msg"); if (err) err.remove();
  area.appendChild(document.getElementById("outputEmpty"));
  area.appendChild(content);
  area.appendChild(getPipelineStepsContainer());
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
  document.getElementById("settingTermVocabulary").value = (settings.termVocabulary || []).join("\n");
  document.getElementById("settingMacros").value = (settings.macros || []).map(m => `${m.key}: ${m.value}`).join("\n");
  document.getElementById("settingCleanupPrompt").value = settings.cleanupPrompt;
  document.getElementById("settingMainPrompt").value = settings.mainPrompt;
  renderPipelineStepsList();
  renderPipelineLibrary();
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

// ─── Extra Pipeline Steps UI ──────────────────────────────────────────────
function makeStepId(prefix = "custom") {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
function renderPipelineStepsList() {
  const container = document.getElementById("pipelineStepsList");
  if (!container) return;
  container.innerHTML = "";
  (settings.pipelineSteps || []).forEach((step, idx) => container.appendChild(createPipelineStepEl(step, idx)));
  container.addEventListener("dragover", handlePipelineDragOver);
  container.addEventListener("drop", handlePipelineDrop);
}
function createPipelineStepEl(step, idx) {
  const div = document.createElement("div");
  div.className = "pipeline-entry";
  div.dataset.idx = idx;
  div.draggable = true;
  div.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("text/plain", String(idx));
    e.dataTransfer.effectAllowed = "move";
    div.classList.add("dragging");
  });
  div.addEventListener("dragend", () => div.classList.remove("dragging"));
  div.innerHTML = `
    <div class="pipeline-entry-header" onclick="togglePipelineEntry(this)">
      <span class="pipeline-drag-handle" title="Drag to reorder" onclick="event.stopPropagation()">⋮⋮</span>
      <label class="switch" onclick="event.stopPropagation()">
        <input type="checkbox" ${step.enabled ? "checked" : ""} onchange="updatePipelineStep(${idx},'enabled',this.checked)">
        <span class="switch-slider"></span>
      </label>
      <input class="pipeline-label-input" type="text" value="${escHtml(step.label)}" placeholder="Step label" spellcheck="false" onclick="event.stopPropagation()" oninput="updatePipelineStep(${idx},'label',this.value)">
      <span class="pipeline-entry-source">${step.inputSource === "cleanedTranscript" ? "Cleaned transcript" : "Note output"}</span>
      <span class="bp-entry-toggle"><svg class="bp-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></span>
    </div>
    <div class="pipeline-entry-body">
      <label class="bp-field-label" for="pipelineInputSource${idx}">Input source</label>
      <select class="settings-select pipeline-source-select" id="pipelineInputSource${idx}" onchange="updatePipelineStep(${idx},'inputSource',this.value)">
        <option value="noteOutput" ${step.inputSource === "noteOutput" ? "selected" : ""}>Structured note output</option>
        <option value="cleanedTranscript" ${step.inputSource === "cleanedTranscript" ? "selected" : ""}>Cleaned transcript</option>
      </select>
      <div>
        <div class="bp-field-label">Prompt</div>
        <textarea class="pipeline-prompt-textarea" rows="10" spellcheck="false" oninput="updatePipelineStep(${idx},'prompt',this.value)">${escHtml(step.prompt)}</textarea>
      </div>
      <div class="bp-entry-footer">
        <button class="bp-btn-delete" onclick="deletePipelineStep(${idx})"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>Delete</button>
      </div>
    </div>`;
  return div;
}
function handlePipelineDragOver(e) {
  e.preventDefault();
  e.dataTransfer.dropEffect = "move";
}
function handlePipelineDrop(e) {
  e.preventDefault();
  const from = Number(e.dataTransfer.getData("text/plain"));
  if (!Number.isInteger(from)) return;
  const rows = [...document.querySelectorAll("#pipelineStepsList .pipeline-entry")];
  const target = e.target.closest(".pipeline-entry");
  let to = target ? rows.indexOf(target) : settings.pipelineSteps.length - 1;
  if (to < 0) to = settings.pipelineSteps.length - 1;
  const targetRect = target?.getBoundingClientRect();
  if (targetRect && e.clientY > targetRect.top + targetRect.height / 2) to += 1;
  if (from < to) to -= 1;
  if (from === to || from < 0 || from >= settings.pipelineSteps.length) return;
  const [moved] = settings.pipelineSteps.splice(from, 1);
  settings.pipelineSteps.splice(Math.max(0, to), 0, moved);
  renderPipelineStepsList();
}
function renderPipelineLibrary() {
  const grid = document.getElementById("pipelineLibraryGrid");
  if (!grid) return;
  grid.innerHTML = "";
  [...PIPELINE_LIBRARY].sort((a,b) => a.order - b.order).forEach(template => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "pipeline-library-card";
    card.innerHTML = `
      <span class="pipeline-library-category">${escHtml(template.category)}</span>
      <strong>${escHtml(template.title)}</strong>
      <span>${escHtml(template.description)}</span>
      <em>${escHtml(template.specialty)}</em>
    `;
    card.addEventListener("click", () => addPipelineStepFromTemplate(template));
    grid.appendChild(card);
  });
}
window.togglePipelineEntry = (h) => h.closest(".pipeline-entry").classList.toggle("expanded");
window.updatePipelineStep = (i, f, v) => {
  if (!settings.pipelineSteps[i]) return;
  settings.pipelineSteps[i][f] = v;
  if (f === "inputSource") renderPipelineStepsList();
};
window.deletePipelineStep = (i) => {
  settings.pipelineSteps.splice(i, 1);
  renderPipelineStepsList();
};
window.addPipelineStep = function() {
  settings.pipelineSteps.push({
    id: makeStepId(),
    label: "Untitled Step",
    enabled: false,
    prompt: "",
    inputSource: "noteOutput"
  });
  renderPipelineStepsList();
  const last = document.getElementById("pipelineStepsList").lastElementChild;
  if (last) { last.classList.add("expanded"); last.querySelector(".pipeline-label-input")?.focus(); }
};
window.openPipelineLibrary = function() {
  renderPipelineLibrary();
  document.getElementById("pipelineLibraryModal").classList.add("open");
};
window.closePipelineLibrary = function() {
  document.getElementById("pipelineLibraryModal").classList.remove("open");
};
window.addPipelineStepFromTemplate = function(template) {
  settings.pipelineSteps.push({
    id: makeStepId(template.id),
    label: template.label,
    enabled: true,
    prompt: template.prompt,
    inputSource: template.inputSource
  });
  renderPipelineStepsList();
  closePipelineLibrary();
  const last = document.getElementById("pipelineStepsList").lastElementChild;
  if (last) last.classList.add("expanded");
};
window.saveSettings = function() {
  settings.llmModel      = document.getElementById("settingLLMModel").value;
  settings.whisperModel  = document.getElementById("settingWhisperModel").value;
  settings.termVocabulary = document.getElementById("settingTermVocabulary").value
    .split("\n").map(s => s.trim()).filter(Boolean);
  
  settings.macros = document.getElementById("settingMacros").value
    .split("\n")
    .map(line => {
      const idx = line.indexOf(":");
      if (idx !== -1) {
        return {
          key: line.slice(0, idx).trim().toLowerCase(),
          value: line.slice(idx + 1).trim()
        };
      }
      return null;
    })
    .filter(m => m && m.key && m.value);

  settings.cleanupPrompt = document.getElementById("settingCleanupPrompt").value;
  settings.mainPrompt    = document.getElementById("settingMainPrompt").value;
  settings.pipelineSteps = normalizePipelineSteps(settings.pipelineSteps);
  const ok = saveSettingsToStorage(settings);
  const status = document.getElementById("settingsSaveStatus");
  status.textContent = ok ? "\u2713 Saved \u2014 click Reload Models to apply model changes" : "\u26a0 Could not persist (storage blocked)";
  status.classList.add("visible");
  setTimeout(() => status.classList.remove("visible"), 3500);
};
window.resetSettings = function() {
  if (!confirm("Reset all settings to defaults? This cannot be undone.")) return;
  settings = structuredClone(DEFAULTS);
  saveSettingsToStorage(settings);
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
initMacroExpanders();
initModel();
