  # Janaswasthya Odisha

  Janaswasthya Odisha is an open-source, multilingual public-health intelligence platform built for Odisha. It combines online health-information
  collection, geographic disease visualisation, environmental analysis, predictive modelling and a grounded AI assistant in one application.

  The platform addresses three primary objectives:

  1. Collect health-related information in Odia, Hindi and English.
  2. Display district-level disease patterns and heatmaps across Odisha.
  3. Estimate one-to-three-month malaria surveillance risk using historical and environmental information.

  ## What it does

  ### Multilingual health intelligence

  The platform maintains 170 acquisition routes across 65 government, health and media hosts. It can process:

  - English, Hindi and Odia web pages
  - Government notices and health bulletins
  - Linked documents and scanned PDFs
  - OCR content using English, Hindi and Odia language models

  Collected information passes through language detection, OCR, translation, disease extraction, assertion classification, district resolution, cross-source
  deduplication and provenance tracking.

  ### Disease and environmental maps

  The interactive dashboard covers all 30 Odisha districts and provides separate layers for:

  - Live health-related evidence collected from online sources
  - Official historical malaria observations
  - District-level malaria surveillance indicators
  - Rainfall, temperature and environmental conditions
  - One, two and three-month predictive outlooks

  Every displayed record retains its source, collection time and evidence status.

  ### Predictive analysis

  The forecasting pipeline combines historical malaria surveillance indicators with seasonality, rainfall and temperature features.

  Models are evaluated using rolling historical validation, calibration diagnostics, environmental-feature ablation and baseline comparisons. Results are
  presented as district-level surveillance-priority probabilities rather than unsupported case-count claims.

  ### Multilingual AI assistant

  The local assistant answers questions about collected evidence, district patterns, environmental conditions and predictive outlooks.

  It supports English, Hindi and Odia using open-source components for:

  - Indic-language translation
  - Multilingual semantic retrieval
  - Grounded answer generation
  - Source and evidence attribution

  No commercial AI API is required.

  ## How it works

  ```text
  Registered sources
          ↓
  Crawler and document acquisition
          ↓
  OCR, language detection and translation
          ↓
  Disease, location and assertion extraction
          ↓
  Deduplication, provenance and review workflow
          ↓
  District heatmaps and environmental features
          ↓
  Predictive models
          ↓
  Grounded multilingual assistant

  The system uses specialised collection, extraction, verification, geographic, modelling and assistant components coordinated as an agentic workflow.
  ```
  
  ## Technology

  - Python, FastAPI and Pydantic
  - React, TypeScript and Vite
  - SQLite locally, with PostgreSQL support
  - Tesseract OCR and Poppler
  - IndicTrans2
  - Multilingual-E5 retrieval
  - Qwen local language model
  - Scikit-learn and gradient-boosted models
  - Docker and Docker Compose

  ## Run locally

  Requirements:

  - Docker Desktop
  - At least 8 GB RAM
  - Approximately 15–20 GB of free disk space

  Download and extract the ZIP, or clone the repository:

  git clone https://github.com/SamparkBhol/health-hub.git
  cd health-hub

  Build the application:

  docker compose build api

  Download the open-source language and assistant models:

  docker compose --profile setup run --rm model-init

  Start the complete platform:

  docker compose up -d

  Wait until the API becomes healthy:

  docker compose ps

  Then open:

  http://localhost:5173

  Stop the platform with:

  docker compose down

  ## Suggested evaluation

  - Open Disease Map to inspect district-level disease patterns.
  - Open Forecast to compare one, two and three-month outlooks.
  - Open Sources to inspect multilingual acquisition routes and evidence.
  - Ask the assistant:

  Which Odisha districts have the highest malaria surveillance priority over the next three months?

  How are rainfall and temperature affecting malaria risk?

  ଓଡ଼ିଶାର କେଉଁ ଜିଲ୍ଲାରେ ମ୍ୟାଲେରିଆ ନିରୀକ୍ଷଣ ପ୍ରାଥମିକତା ଅଧିକ?

  अगले तीन महीनों में ओडिशा के किन जिलों में मलेरिया निगरानी की प्राथमिकता अधिक है?

  ## Live evaluation

  The complete hosted stack runs through GitHub Codespaces and requires approximately 10–15 minutes to activate its crawler, models, database and public
  endpoint.

  Please contact me before evaluating the live deployment. I will activate the Codespace and provide the working public link directly.
