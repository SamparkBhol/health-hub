---
title: Janaswasthya Odisha
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
fullWidth: true
header: mini
---

# Janaswasthya Odisha

Janaswasthya is a multilingual public-health intelligence system for Odisha. It
collects health-related information from registered Odia, Hindi and English
sources, extracts district-level evidence, displays disease-pattern maps and
serves a one-to-three-month malaria research outlook through a local assistant.

Everything runs with open-source software and public data. Translation,
retrieval and answer generation run inside the application; no commercial AI
API is required.

## What works

### Multilingual collection

The source registry contains 170 acquisition routes across 65 hosts. The live
path handles HTML, linked documents and scanned PDFs using bounded fetching,
Poppler and Tesseract OCR (`ori`, `hin`, `eng`). It performs script detection,
disease/place/assertion extraction, cross-source deduplication and review-state
assignment.

### District maps

The interface keeps unlike measurements in separate layers:

- live published-evidence patterns from the multilingual collector;
- official NCVBDC annual district malaria observations for 2010–2024;
- Odisha HMIS district-month malaria indicators for April 2012–March 2020;
- current rainfall, temperature and environmental context for all 30 districts.

Missing districts remain unknown rather than being converted to zero. District
geometry is the DataMeet Census 2011 Odisha set under CC BY 2.5 India.

### Predictive analysis

The public model estimates whether monthly malaria microscopy positivity will
exceed each district's historical 75th-percentile level over the next one, two
or three months. It uses recent malaria positivity, calendar season, rainfall
and temperature outlooks, with the latest official annual malaria burden shown
as a separate priority rank.

The model was evaluated with rolling origins in 2017, 2018 and 2019, seven
competitors, calibration diagnostics and an environmental-feature ablation.
The selected ridge model has Brier score `0.06518`, AUC `0.849` and Brier skill
`+0.425767` against the unconditional constant.

The probability describes an elevated HMIS surveillance indicator, not a
guarantee that a clinically confirmed outbreak will occur.

### Local multilingual assistant

| Function | Runtime |
|---|---|
| English ↔ Odia/Hindi translation | IndicTrans2 with CTranslate2 |
| Cross-language evidence retrieval | multilingual-E5-small ONNX |
| Grounded evidence answers | Qwen2.5-1.5B-Instruct GGUF with llama.cpp |
| Scanned document OCR | Tesseract Odia, Hindi and English |

Official-statistics and forecast questions are answered from validated data
structures. Free-form evidence questions use retrieval and grounded generation
and return supporting record identifiers.

## Run the complete system locally

Requirements:

- Docker Desktop with Docker Compose v2;
- approximately 5 GB available RAM;
- approximately 12 GB free disk;
- a short Windows path such as `C:\health-hub`.

In PowerShell, from the extracted project:

```powershell
Set-Location C:\Users\skbho\Downloads\health-hub
Copy-Item .env.example .env -ErrorAction SilentlyContinue

docker compose build api

$API_IMAGE = docker compose images -q api
docker run --rm `
  -e ODISHA_MODELS_DIR=/app/models `
  -v "${PWD}/models:/app/models" `
  --entrypoint python `
  $API_IMAGE `
  /app/scripts/fetch_models.py

docker run --rm `
  -e ODISHA_MODELS_DIR=/app/models `
  -v "${PWD}/models:/app/models" `
  --entrypoint python `
  $API_IMAGE `
  /app/scripts/fetch_models.py --check

docker compose up -d
docker compose ps
```

Wait until `api` shows `healthy`, then open:

```powershell
Start-Process http://localhost:5173
```

Useful local endpoints:

- interface: <http://localhost:5173>
- API readiness: <http://localhost:8000/api/v1/readyz>
- interactive API: <http://localhost:8000/docs>
- collection state: <http://localhost:8000/api/v1/collector/status>

Stop without deleting downloaded models:

```powershell
docker compose down
```

## Free public deployment from GitHub

The complete stack uses about 4.3 GB RAM when crawling, translating, retrieving
and generating together. It is packaged as one Hugging Face Docker Space:
FastAPI, the React interface, continuous collection, environmental refresh,
IndicTrans2, multilingual-E5 and Qwen all run in the same free CPU container.

### 1. Push this repository

```powershell
Set-Location C:\Users\skbho\Downloads\health-hub
git init
git add .
git commit -m "Initial Janaswasthya release"
git branch -M main
git remote add origin https://github.com/SamparkBhol/health-hub.git
git push -u origin main
```

### 2. Create the free Space

1. Sign in at <https://huggingface.co/join>.
2. Open <https://huggingface.co/new-space>.
3. Enter `health-hub` as the Space name.
4. Choose a public Space.
5. Select **Docker** as the SDK.
6. Select the free **CPU Basic** hardware.
7. Create the Space.

No model-provider API key is required by the application.

### 3. Connect GitHub to the Space

1. In Hugging Face, open **Settings → Access Tokens → New token**.
2. Create a fine-grained write token scoped to the `health-hub` Space.
3. In GitHub, open `SamparkBhol/health-hub`.
4. Open **Settings → Secrets and variables → Actions → Secrets**.
5. Add a repository secret named `HF_TOKEN`; paste the Hugging Face token.
6. Open the **Variables** tab on the same page.
7. Add `HF_SPACE_ID` with value `YOUR_HF_USERNAME/health-hub`.
8. Open GitHub **Actions → sync hugging face space**.
9. Select **Run workflow → Run workflow**.

GitHub uploads the source to the Space. Hugging Face then builds the Docker
image, downloads the four public model packages into the image and starts the
application. The first build is large and can take 20–45 minutes. Follow the
Space's **Building** log until its status becomes **Running**.

The public URL is displayed on the Space page and normally has this form:

```text
https://YOUR_HF_USERNAME-health-hub.hf.space
```

Every later push to GitHub `main` automatically synchronises and rebuilds the
Space.

Free CPU Spaces sleep after inactivity. Visiting the URL wakes the same
application again. Runtime crawler records are ephemeral on the free disk and
are reacquired after a restart; the official maps, trained model and model
weights are built into the image.

## Questions that exercise the complete system

Ask these in the **Assistant** tab:

```text
Which district had more malaria cases in 2024, Koraput or Malkangiri?
```

```text
Show me the malaria heatmap across Odisha.
```

```text
Predict malaria risk in Kandhamal for the next 3 months.
```

```text
Which districts have the highest likelihood of an elevated malaria indicator next month?
```

```text
Which districts have published dengue evidence, and from which sources?
```

```text
Show dengue evidence for Khordha and cite the supporting records.
```

```text
ଗଞ୍ଜାମରେ ଡେଙ୍ଗୁ ସତର୍କ ସଙ୍କେତ ଦେଖାନ୍ତୁ।
```

```text
खुर्दा जिले में डेंगू के बारे में कौन से रिकॉर्ड प्रकाशित हुए हैं?
```

```text
ଖୋର୍ଦ୍ଧାରେ ମ୍ୟାଲେରିଆ ସମ୍ଭାବନା ଆଗାମୀ ୩ ମାସରେ କେତେ?
```

```text
Can the EpiClim historical data train an Odisha outbreak forecast?
```

## Main API routes

| Route | Purpose |
|---|---|
| `GET /api/v1/sources` | Source registry and acquisition state |
| `GET /api/v1/maps/published-signals` | Live evidence-pattern map |
| `GET /api/v1/public-health/malaria/map` | Official annual malaria map |
| `GET /api/v1/public-health/hmis/map` | Historical HMIS indicator map |
| `GET /api/v1/environment/current/map` | Current environmental map |
| `GET /api/v1/outlook/public/map` | One-to-three-month malaria outlook |
| `GET /api/v1/outlook/public/evaluation` | Backtest and ablation evidence |
| `POST /api/v1/translate` | Odia/Hindi/English translation |
| `POST /api/v1/agent/query` | Grounded assistant |

## Engineering verification

```bash
uv sync --all-groups --all-extras
npm --prefix apps/web ci
make verify
```

Source policies and attribution are recorded in [SOURCES.md](SOURCES.md),
[NOTICE.md](NOTICE.md) and the executable source registry. The software is
licensed under Apache-2.0.
