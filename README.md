# ClinicalScribe

**AI-powered Assessment & Plan documentation tool that runs entirely in your browser.**

Dictate or type your clinical assessment and plan — ClinicalScribe converts it into concise, telegraphic structured notes per problem.

## Features

- 🎙️ **Voice input** — speak your A&P dictation using the Web Speech API
- ⌨️ **Text input** — paste or type dictation
- 🤖 **On-device AI** — powered by [WebLLM](https://webllm.mlc.ai/) (Qwen2.5-1.5B), runs 100% in the browser
- 🔒 **No data transmission** — nothing leaves your device
- 📋 **Copy to clipboard** — one-click copy of structured output

## Output Format

For each problem/diagnosis, the tool produces only the relevant bullets:

```
Diagnosis
- Differential includes X, Y, Z
- Labs
- Imaging
- Medications / Treatment
- Supportive Care
- Situational Awareness / Conditional Orders
- Return Precautions
- Nursing Orders
- Follow-Up
```

Bullets not mentioned in dictation are omitted.

## Usage

1. Open the app in **Chrome 113+** (WebGPU required)
2. Wait for the model to load (~700MB download, cached after first load)
3. Choose Voice or Text input
4. Speak or type your assessment and plan
5. Click **Generate Notes**
6. Copy the output

## Deployment (GitHub Pages)

1. Fork or clone this repo
2. Go to **Settings → Pages**
3. Set source to `main` branch, `/ (root)` folder
4. Save — your site will be live at `https://[username].github.io/[repo-name]/`

No build step required — this is a pure static site.

## Browser Requirements

- **Chrome 113+** or **Edge 113+** (WebGPU support required)
- WebGPU may need to be enabled via `chrome://flags/#enable-unsafe-webgpu` on some versions
- Safari and Firefox do not yet support WebGPU

## Tech Stack

- [WebLLM](https://github.com/mlc-ai/web-llm) — in-browser LLM inference
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — language model
- Web Speech API — browser-native speech recognition
- Vanilla HTML/CSS/JS (ES modules)

---

*For clinical documentation assistance only. Not a substitute for clinical judgment.*
