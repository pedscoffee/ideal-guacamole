# Present - AI-Assisted A&P Notes

*This tool is currently in beta development. The goal is to explore what is possible, but in its current state any actual clinical use should be strictly avoided.*

> Runs entirely in your browser. Clinical dictation, transcripts, prompts, and generated notes are processed locally on the device.

[Live App](https://pedscoffee.github.io/ideal-guacamole/)

Present converts spoken or typed clinical assessment and plan dictation into clean, structured, telegraphic notes, using on-device speech-to-text and local browser-based LLM inference.

For patient-data and IT review details, see [SECURITY.md](SECURITY.md).

---

## Features

- Voice input: record A&P dictation; Whisper transcribes locally in the browser.
- Text input: paste or type dictation directly.
- Editable transcript: correct transcription issues before note generation.
- Two-pass AI pipeline: pass 1 cleans rough dictation; pass 2 generates structured note blocks.
- Selectable local models: choose LLM and Whisper model size based on hardware.
- Medical terminology anchoring: add practice-specific terms that guide transcript cleanup.
- Shorthand macros: expand dot phrases like `.aom` or `.rtp` while typing and before generation.
- Optional extra pipeline steps: generate AVS text, billing analysis, teaching prompts, or custom outputs.
- Smart boilerplate injection: condition-specific paragraphs are appended based on generated tags.
- Editable output: revise the final note before pasting into the EMR.
- Auto-copy and manual copy: copy the full note or individual problem blocks.
- Local settings: prompts, boilerplate, models, terminology, macros, and pipeline steps persist in `localStorage`.
- Installable PWA: app shell, dependencies, and fetched model assets are cached for offline local use after first load.
- Static deployment: no backend, no database, no account system, and no cloud AI API.

---

## Patient Data Posture

Present is designed for local/on-device clinical documentation:

- Audio capture uses the browser microphone API.
- Transcription runs locally with Transformers.js and Whisper.
- Note generation runs locally with WebLLM and WebGPU.
- Patient text/audio is not intentionally sent to an external AI service or application backend.
- Generated note content is held in the browser session and copied to the clipboard for clinician use.

The default hosted app loads JavaScript dependencies and model assets from third-party static origins during first setup. After those assets are fetched, the PWA service worker and browser caches allow local offline use. For stricter institutional review, deploy a self-hosted version where app files, dependencies, and model assets are served from an approved institution-controlled origin.

---

## Usage

1. Open the app in Chrome 113+ or Edge 113+.
2. Wait for the LLM and Whisper models to load. Model files are large and are cached by the browser after first load.
3. Choose Voice or Text input.
4. Voice: click the mic, speak the A&P, stop recording, then edit the transcript if needed.
5. Text: paste or type the dictation.
6. Click Generate Notes.
7. Review and edit the structured output.
8. Paste into the EMR, or copy individual problem blocks as needed.

Always review AI-generated notes before use in the medical record.

## PWA / Offline Local Use

Present can be installed as a local PWA from a trusted HTTPS origin or from `localhost`.

1. Open the app while online.
2. Let the selected LLM and Whisper models finish loading.
3. Open Settings -> Local Security -> Check offline cache.
4. Install the app from the browser toolbar/menu.
5. Disconnect from the network and reopen the installed app to confirm it starts locally.

The app shell is pre-cached by `sw.js`. Runtime dependencies and model files are cached as they are requested from the approved static model/dependency hosts. If you switch to a different model, reconnect once and let that model finish loading before relying on offline use.

Security notes:

- Use HTTPS or `localhost`; browsers require a secure context for service workers, microphone access, and WebGPU.
- The service worker does not add any backend, analytics, or remote logging.
- Patient note text is still transient page memory and clipboard content; it is not intentionally written to `localStorage`.
- Browser extensions, OS clipboard managers, endpoint security posture, and device access controls remain institution responsibilities.

---

## Output Format

For each problem or diagnosis the clinician mentions, only relevant bullets are emitted:

```text
Diagnosis or Problem Name
- Labs
- Imaging
- Medications with exact doses if stated
- Treatment / plan actions
- Supportive care
- Differential includes X, Y, Z
- Conditional plans
- Return precautions include...
- Nursing orders
- Follow-Up: ...

[Condition-appropriate boilerplate appended automatically]
```

Bullets not mentioned in the dictation are omitted.

---

## Copying Notes

The output panel supports several copy workflows:

- Copy All copies the full note, including boilerplate.
- Per-problem copy appears on each problem block.
- The Copy All dropdown lists each problem so a specific block can be copied without scrolling.
- Auto-copy copies the completed note shortly after generation.

The output remains editable after generation, so clinicians can correct wording before copying.

---

## Settings

Open the gear icon in the top-right header to access settings.

| Setting | Description |
|---|---|
| Language Model | Selects the local WebLLM model used for cleanup, note generation, and optional pipeline steps. |
| Whisper Speech-to-Text Model | Selects the local transcription model. Larger models are more accurate but require more memory and download time. |
| Medical Terminology | Adds practice-specific vocabulary used during transcript cleanup. |
| Shorthand Macros | Defines dot phrases such as `.aom: acute otitis media`. |
| Pass 1 - Transcript Cleanup Prompt | System prompt used to clean raw ASR or typed dictation. |
| Pass 2 - Note Generation Prompt | System prompt used to generate structured A&P notes. |
| Extra Pipeline Steps | Optional post-processing outputs such as AVS/sign-off, billing analysis, teaching prompts, or custom steps. |
| Boilerplate Entries | Condition-specific text blocks inserted when matching tags are generated. |

Click Save & Apply to persist settings. Model changes require Reload Models. Reset defaults restores factory prompts, boilerplate, terminology, macros, models, and pipeline settings.

Settings are stored in browser `localStorage`; avoid placing patient identifiers in persistent settings.

---

## Model Options

Current LLM options:

| Option | Approx. size | Notes |
|---|---:|---|
| Gemma 2 2B | 1.3 GB | Fastest option. |
| Phi 3.5 Mini | 2.2 GB | Balanced-light option. |
| Qwen3 4B | 2.3 GB | Default balanced option. |
| Llama 3 8B | 4.5 GB | Higher quality; requires more memory. |

Current Whisper options:

| Option | Approx. size | Notes |
|---|---:|---|
| Whisper Tiny | 75 MB | Fastest, lower accuracy. |
| Whisper Base | 145 MB | Lightweight balance. |
| Whisper Small | 488 MB | Default recommended option. |
| Whisper Medium | 1.5 GB | Best accuracy; requires more memory. |

---

## Extra Pipeline Steps

Extra pipeline steps run after the main note is generated. Each enabled step produces its own output box with a separate copy button.

Built-in templates include:

- AVS / Sign-Off: generates personalized sign-off options and a family to-do list.
- Billing Analysis: estimates established-patient E/M coding using 2021 MDM rules.
- Teaching - Socratic: extracts a clinical pearl and follow-up teaching question.

Custom steps can use either the cleaned transcript or the generated note as input.

---

## Boilerplate Tags

The LLM emits structured tags when conditions are present; the app replaces them with clinician-reviewed boilerplate text at render time.

| Tag | Trigger condition |
|---|---|
| `[BOILERPLATE:WCC]` | Well child check / health maintenance |
| `[BOILERPLATE:ILLNESS]` | Illness, infection, virus, fever |
| `[BOILERPLATE:INJURY]` | Injury |
| `[BOILERPLATE:OTITIS]` | Otitis media |
| `[BOILERPLATE:STREP]` | Strep throat / rapid strep |
| `[BOILERPLATE:DEHYDRATION]` | Dehydration, vomiting, diarrhea, decreased urination |
| `[BOILERPLATE:RESP]` | Trouble breathing, wheezing, respiratory distress |
| `[BOILERPLATE:PCMH]` | ADHD, weight concern, obesity, or strep throat |

Boilerplate entries can be edited or added in the settings drawer without changing code.

---

## Deployment

### GitHub Pages

1. Fork this repo.
2. Go to Settings -> Pages.
3. Set source to `main` branch, `/ (root)` folder.
4. Save. The app will be available at `https://[username].github.io/[repo-name]/`.

No build step is required.

### Enterprise / IT Deployment

For institutional review, consider hosting the following from an approved origin:

- `index.html`, `style.css`, `app.js`, and `favicon.svg`.
- `manifest.webmanifest` and `sw.js`.
- WebLLM JavaScript dependency.
- Transformers.js dependency.
- LLM model assets.
- Whisper model assets.

See [SECURITY.md](SECURITY.md) for the data-flow table, network disclosure, PHI handling statement, and suggested IT approval language.

---

## Browser Requirements

| Browser | WebGPU | WASM | Status |
|---|---|---|---|
| Chrome 113+ | Yes | Yes | Fully supported |
| Edge 113+ | Yes | Yes | Fully supported |
| Safari | Partial | Yes | Experimental / may require flags |
| Firefox | No | Yes | Not supported for WebGPU LLM inference |

If WebGPU is disabled, try enabling `chrome://flags/#enable-unsafe-webgpu`.

---

## Tech Stack

| Component | Library / Model |
|---|---|
| In-browser LLM inference | [WebLLM](https://github.com/mlc-ai/web-llm) |
| Default language model | Qwen3 4B MLC variant |
| Speech-to-text | [Transformers.js](https://github.com/huggingface/transformers.js) + Whisper models |
| Frontend | Vanilla HTML, CSS, and JavaScript ES modules |
| Persistence | Browser `localStorage` for settings |
| Acceleration | WebGPU for LLM, WASM for Whisper |

---

## Clinical Disclaimer

Present is for clinical documentation assistance only. It is not a substitute for clinical judgment, billing expertise, legal review, or institutional compliance review. Clinicians are responsible for reviewing generated content before use.
