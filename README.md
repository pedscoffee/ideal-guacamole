# Present — AI-Assisted A&P Notes

> **Runs entirely in your browser. No data ever leaves your device.**

[**Live App →**](https://pedscoffee.github.io/ideal-guacamole/)

Present converts spoken or typed clinical assessment & plan dictation into clean, structured, telegraphic notes — one block per problem — using on-device AI inference.

---

## Features

- 🎙️ **Voice input** — record your A&P dictation; Whisper (Xenova/whisper-tiny.en) transcribes it locally via WebAssembly
- ✏️ **Editable transcript** — after transcription, the transcript box becomes editable so you can correct any mishears before generating notes
- ⌨️ **Text input** — paste or type dictation directly
- 🤖 **Two-pass AI pipeline** — pass 1 cleans the raw ASR transcript; pass 2 generates structured notes (both run on [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) via WebLLM)
- 📋 **Auto-copy** — the finished note is automatically copied to the clipboard ~400 ms after generation, with a toast confirmation
- 📋 **Per-problem copy** — hover any problem block to reveal an inline copy icon; click it to copy just that problem's text — ideal for pasting individual diagnoses into separate Epic note fields
- 📋 **Copy dropdown** — the "Copy All" button in the output header includes a chevron that opens a dropdown listing each problem by name, so you can copy any single problem without scrolling
- ✏️ **Editable output** — the structured note panel is `contenteditable`; make quick tweaks before pasting into your EMR
- 💉 **Smart boilerplate injection** — condition-specific boilerplate paragraphs (illness, otitis media, strep, dehydration, respiratory distress, injury, WCC, PCMH) are appended automatically based on diagnoses detected in the note
- ⚙️ **Settings drawer** — customize both LLM system prompts (Pass 1 transcript cleanup, Pass 2 note generation) and all boilerplate entries (key, trigger phrase, text) without editing code; settings persist via localStorage
- 🔒 **100% on-device** — no server, no API calls, no PHI transmitted anywhere
- ⚡ **WebGPU accelerated** — LLM runs via WebGPU for fast inference; Whisper runs on WASM

---

## Output Format

For each problem or diagnosis the clinician mentions, only the relevant bullets are emitted:

```
Diagnosis or Problem Name
- Labs
- Imaging
- Medications (exact dose if stated)
- Treatment / plan actions
- Supportive care
- Differential includes X, Y, Z
- Conditional plans
- Return precautions include...
- Nursing orders
- Follow-Up: ...

[Condition-appropriate boilerplate appended automatically]
```

Bullets not mentioned in the dictation are omitted entirely.

---

## Copying Notes

### Copy All
The **Copy All** button in the output panel header copies the entire structured note (all problems + boilerplate) to the clipboard. This is the default for single-problem notes or when pasting into a single EMR field.

### Per-Problem Copy
Hover over any individual problem block to reveal a small copy icon to the right of the diagnosis heading. Clicking it copies only that problem's text — useful when pasting each diagnosis into a separate field in Epic's structured note sections.

### Copy Dropdown
Click the **chevron (▾)** next to "Copy All" to open a dropdown that lists every problem by name with a numbered badge. Click any item to copy that single problem. This lets you target a specific diagnosis without needing to hover the block itself.

---

## Usage

1. Open the app in **Chrome 113+** or **Edge 113+** (WebGPU required)
2. Wait for both models to load — **Qwen3-4B (~2.3 GB)** and **Whisper tiny (~75 MB)** — cached by the browser after the first load
3. Choose **Voice** or **Text** input
4. **Voice:** click the mic, speak your A&P, click stop — Whisper transcribes locally; edit the transcript if needed
5. **Text:** paste or type your dictation
6. Click **Generate Notes**
7. The structured note is rendered, auto-copied to clipboard, and is directly editable in the output panel
8. Paste into your EMR — or use per-problem copy / the dropdown to target individual Epic note fields

---

## Settings

Open the **gear icon (⚙)** in the top-right header to access the settings drawer:

| Setting | Description |
|---|---|
| **Pass 1 — Transcript Cleanup Prompt** | System prompt sent to the LLM to clean raw ASR output before structuring |
| **Pass 2 — Note Generation Prompt** | System prompt that structures the cleaned dictation into A&P note blocks |
| **Boilerplate Entries** | Each entry has a key (`[BOILERPLATE:KEY]`), trigger phrases (instructions to the LLM), and boilerplate text injected into the output |

Click **Save & Apply** to persist changes. Use **Reset defaults** to restore factory prompts and boilerplate. Settings are stored in `localStorage` and survive page reloads.

---

## Deployment (GitHub Pages)

1. Fork this repo
2. Go to **Settings → Pages**
3. Set source to `main` branch, `/ (root)` folder
4. Save — the app will be live at `https://[username].github.io/[repo-name]/`

No build step required. Pure static site.

---

## Browser Requirements

| Browser | WebGPU | WASM | Status |
|---|---|---|---|
| Chrome 113+ | ✅ | ✅ | ✅ Fully supported |
| Edge 113+ | ✅ | ✅ | ✅ Fully supported |
| Safari | ⚠️ Partial | ✅ | ⚠️ WebGPU experimental |
| Firefox | ❌ | ✅ | ❌ WebGPU not supported |

If WebGPU is not enabled, try `chrome://flags/#enable-unsafe-webgpu`.

---

## Tech Stack

| Component | Library / Model |
|---|---|
| In-browser LLM inference | [WebLLM](https://github.com/mlc-ai/web-llm) |
| Language model | [Qwen3-4B-q4f16_1-MLC](https://huggingface.co/Qwen/Qwen3-4B) |
| Speech-to-text | [Hugging Face Transformers.js](https://github.com/xenova/transformers.js) + [Xenova/whisper-tiny.en](https://huggingface.co/Xenova/whisper-tiny.en) |
| Frontend | Vanilla HTML / CSS / JS (ES modules, no build step) |
| Fonts | IBM Plex Mono + IBM Plex Sans (Google Fonts) |

---

## Boilerplate Tags

The LLM emits structured tags when conditions are present; the app replaces them with clinician-reviewed boilerplate text at render time. Tags currently supported:

| Tag | Trigger condition |
|---|---|
| `[BOILERPLATE:WCC]` | Well child check / health maintenance |
| `[BOILERPLATE:ILLNESS]` | Any illness, infection, fever |
| `[BOILERPLATE:INJURY]` | Any injury |
| `[BOILERPLATE:OTITIS]` | Otitis media |
| `[BOILERPLATE:STREP]` | Strep throat / rapid strep |
| `[BOILERPLATE:DEHYDRATION]` | Dehydration, vomiting, diarrhea |
| `[BOILERPLATE:RESP]` | Respiratory distress / wheezing |
| `[BOILERPLATE:PCMH]` | ADHD, obesity, strep (PCMH reminder) |

To update boilerplate text or add new entries, use the **Settings drawer** (no code editing required). To add new tags programmatically, edit the `DEFAULTS.boilerplate` array in `app.js`.

---

*For clinical documentation assistance only. Not a substitute for clinical judgment. Always review AI-generated notes before use.*
