# Janaswasthya Odisha

  Janaswasthya Odisha is an open-source, multilingual public-health intelligence platform for Odisha. It combines online health-information collection,
  district-level disease mapping, environmental analysis, predictive modelling and a grounded AI assistant in one application.

  ## Core capabilities

  ### 1. Multilingual health-information collection

  The source registry contains 170 acquisition routes across government, public-health and media websites.

  The collection pipeline supports:

  - Odia, Hindi and English web pages
  - Government notices and health bulletins
  - HTML, PDF and scanned-document ingestion
  - Tesseract OCR for Odia, Hindi and English
  - Language identification and IndicTrans2 translation
  - Disease, location and assertion extraction
  - Cross-source deduplication
  - Source provenance and review status

  The application reports the runtime state of every source instead of treating a configured source as successfully collected.

  ### 2. Disease and environmental maps

  The dashboard covers all 30 districts of Odisha and provides separate map layers for:

  - Live health-related evidence collected from registered sources
  - Official annual malaria observations
  - District-month malaria surveillance indicators
  - Rainfall, temperature and environmental conditions
  - One, two and three-month malaria surveillance outlooks

  Each displayed record retains its source, observation period, collection time and evidence state.

  ### 3. Predictive analysis

  The modelling pipeline combines:

  - Historical malaria surveillance indicators
  - Seasonal patterns
  - Rainfall observations and outlooks
  - Temperature observations and outlooks
  - Recent district-level surveillance behaviour

  Models are evaluated using rolling historical validation, calibration diagnostics, baseline comparisons and environmental-feature ablation.

  The public outlook estimates whether a district’s malaria surveillance indicator is likely to exceed its historical reference level. It does not convert
  missing observations into zero disease or present unsupported case-count forecasts.

  ### 4. Multilingual AI assistant

  The assistant answers questions about:

  - Collected health evidence
  - District disease patterns
  - Environmental conditions
  - Historical malaria observations
  - One-to-three-month surveillance outlooks

  It supports English, Hindi and Odia using IndicTrans2, multilingual semantic retrieval and a local Qwen model. Answers are grounded in application records
  and include evidence context instead of relying on an external commercial AI API.

  ## System workflow

  ```text
  Registered health sources
            │
            ▼
  Crawler and document acquisition
            │
            ▼
  OCR, language detection and translation
            │
            ▼
  Disease, location and assertion extraction
            │
            ▼
  Deduplication, provenance and review workflow
            │
            ▼
  District maps and environmental features
            │
            ▼
  Predictive modelling
            │
            ▼
  Grounded multilingual assistant
  ```

  The collection, extraction, verification, geographic, modelling and assistant components operate as a coordinated agentic workflow. Each stage has
  explicit inputs, outputs, failure states and provenance.

  ## Technology

  - Python, FastAPI and Pydantic
  - React, TypeScript and Vite
  - SQLite for local execution, with PostgreSQL support
  - Tesseract OCR and Poppler
  - IndicTrans2
  - Multilingual-E5
  - Qwen2.5 with llama.cpp
  - Scikit-learn and gradient-boosted models
  - Docker and Docker Compose


<img width="1886" height="948" alt="image" src="https://github.com/user-attachments/assets/19f13594-d68f-404c-ae27-a494d284a7b9" />

<img width="1741" height="683" alt="image" src="https://github.com/user-attachments/assets/95e6a1ec-cd53-4607-ac36-148d7c31f935" />

<img width="1580" height="852" alt="image" src="https://github.com/user-attachments/assets/8fb3bfd3-ffbe-4515-9d70-f47b676a80af" />


  ## Run locally

  ### Requirements

  - Docker Desktop
  - Git
  - At least 8 GB RAM
  - Approximately 20 GB of free disk space

  ### 1. Download the project

  ```bash
  git clone https://github.com/SamparkBhol/health-hub.git
  cd health-hub
  ```

  Alternatively, download the repository ZIP, extract it and open the extracted `health-hub` directory in a terminal.
  ```bash
  docker compose build api
  ```

  ### 3. Download the local AI models

  The model weights are not stored in Git because they require several gigabytes.

  On Linux, macOS or Git Bash:

  ```bash
  mkdir -p models

  API_IMAGE="$(docker compose images -q api)"

  docker run --rm \
    --user root \
    -e ODISHA_MODELS_DIR=/app/models \
    -v "$PWD/models:/app/models" \
    --entrypoint python \
    "$API_IMAGE" \
    /app/scripts/fetch_models.py
  ```

  On Windows PowerShell:

  ```powershell
  New-Item -ItemType Directory -Force .\models | Out-Null

  $MODEL_DIR = (Resolve-Path .\models).Path
  $API_IMAGE = (docker compose images -q api).Trim()

  docker run --rm `
    --user root `
    -e ODISHA_MODELS_DIR=/app/models `
    -v "${MODEL_DIR}:/app/models" `
    --entrypoint python `
    $API_IMAGE `
    /app/scripts/fetch_models.py
  ```

  ### 4. Start the application

  ```bash
  docker compose up --build -d
  ```

  Check the services:

  ```bash
  docker compose ps
  ```

  Wait until the API reports `healthy`, then open:

  ```text
  http://localhost:5173
  ```

  ### 5. Stop the application

  ```bash
  docker compose down
  ```

  ## Suggested evaluation

  Open the following sections:

  - **Disease Map** — inspect district-level malaria and collected-evidence layers.
  - **Forecast** — compare one, two and three-month district outlooks.
  - **Sources** — inspect acquisition routes, languages and runtime states.
  - **Assistant** — ask questions in English, Hindi or Odia.

  Example questions:

  ```text
  Which Odisha districts have the highest malaria surveillance priority over the next three months?
  ```

  ```text
  How are rainfall and temperature affecting the malaria outlook?
  ```

  ```

  ```text
  अगले तीन महीनों में ओडिशा के किन जिलों में मलेरिया निगरानी की प्राथमिकता अधिक है?
  ```

  ## Verification

  Run the automated validation suite with:

  ```bash
  make verify
  ```

  The verification pipeline covers ingestion, OCR integration, trilingual extraction, geographic resolution, forecasting, API behaviour, workflow
  transitions and frontend compilation.

  ## Live evaluation

  The complete hosted stack runs through GitHub Codespaces. Activating its crawler, database, local language models and public endpoint requires
  approximately 10–15 minutes.

  **Please contact me before evaluating the live deployment. I will activate the Codespace and provide the working public link directly.**
