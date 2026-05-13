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
- ✏️ **Editable output** — the structured note panel is `contenteditable`; make quick tweaks before pasting into your EMR
- 💉 **Smart boilerplate injection** — condition-specific boilerplate paragraphs (illness, otitis media, strep, dehydration, respiratory distress, injury, WCC, PCMH) are appended automatically based on diagnoses detected in the note
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

## Usage

1. Open the app in **Chrome 113+** or **Edge 113+** (WebGPU required)
2. Wait for both models to load — **Qwen3-4B (~2.3 GB)** and **Whisper tiny (~75 MB)** — cached by the browser after the first load
3. Choose **Voice** or **Text** input
4. **Voice:** click the mic, speak your A&P, click stop — Whisper transcribes locally; edit the transcript if needed
5. **Text:** paste or type your dictation
6. Click **Generate Notes**
7. The structured note is rendered, auto-copied to clipboard, and is directly editable in the output panel
8. Paste into your EMR

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

To update boilerplate text, edit the `BOILERPLATE` object in `app.js`.

---

*For clinical documentation assistance only. Not a substitute for clinical judgment. Always review AI-generated notes before use.*
