# Trumpet AI Teaching Lab

A multimodal AI teaching system for trumpet pedagogy, combining retrieval-augmented generation, audio workflows, and structured evaluation.

> **Portfolio snapshot.** This public repository is curated to demonstrate engineering work. It is **source-available for review only (not open source)** and is **not a runnable clone of the full teaching product**. The private instructional corpus, vector index, production prompts, proprietary rubrics, and the complete teaching ("lesson") logic are intentionally excluded — see [`docs/portfolio-snapshot.md`](docs/portfolio-snapshot.md) and [License](#license).

The teaching focus is **Jerome Callet-inspired trumpet pedagogy** (sound production, embouchure coordination, efficiency, register development). This project is **not an official Callet resource**.

## Overview

- **Knowledge mode** (`/chat`) — grounded Q&A over curated instructional material, streamed, with tiered retrieval and per-era prompt assembly.
- **Audio analysis** — a librosa-based analysis path plus feature-extraction skeletons for articulation / register / tone work (the proprietary scoring rubrics are withheld in this snapshot).
- The stateful lesson/teaching product is **excluded** from this snapshot.

## Engineering Highlights

- Domain-specific AI assistant rather than a generic chatbot.
- Grounds responses in curated context with a **tiered retrieval budget** (not naive top-K).
- Clean separation of source material, AI interpretation, and (in the full system) human-approved guidance.
- Privacy/rights-aware design: private corpus separated from public sample data, with git history scrubbed.

## What this demonstrates

- Domain-specific RAG over curated instructional material
- Prompt and retrieval design for a specialized teaching domain
- Audio/DSP and transcription workflow engineering
- Structured (LLM-as-judge) evaluation of responses
- Privacy- and rights-aware handling of instructional assets
- Practical FastAPI / Python AI product engineering

## System Architecture

See [`docs/architecture.md`](docs/architecture.md) for a diagram. Components:

- **API / backend** — FastAPI (`teaching-engine/app`): streaming `/chat`, audio, health, and session routes.
- **Retrieval / indexing** — tiered character-budget retrieval over ChromaDB with sentence-transformer embeddings, era-filtered at query time. *The index built from the private corpus is not included.*
- **Prompt construction** — shared + per-era prompt layers assembled at request time, with retrieved context injected as cited source blocks. *(Production prompt content withheld; skeletons show the structure.)*
- **Audio / DSP** — librosa-based feature extraction; scoring rubrics withheld.
- **Evaluation** — an LLM-as-judge harness scoring grounding, alignment, completeness, and helpfulness.

## Current Status

**In this snapshot (runnable / inspectable):**

- Grounded Q&A endpoint (`/chat`) with the tiered retrieval pipeline
- Prompt-assembly architecture (shared + per-era layers) via skeleton prompts
- Audio feature-extraction skeletons (DSP) + a generic librosa analyzer
- LLM-as-judge evaluation harness (code + synthetic examples)
- Mock mode for no-cost local runs and tests over synthetic data

**Excluded from this snapshot** (proprietary / private — see [`docs/portfolio-snapshot.md`](docs/portfolio-snapshot.md)):

- The stateful lesson/teaching product (curriculum, lesson-track composition, proficiency, drills, placement, orchestration)
- Production system prompts and Callet-specific teaching logic
- Diagnostic scoring rubrics / calibration thresholds
- The private source corpus, recordings, transcripts, and the built vector index
- The ingestion/scraping/vision pipeline and internal tooling

This is a research/prototype snapshot, not production software.

## Quickstart

```bash
cd teaching-engine
python -m venv .venv
# core runtime deps (no unified requirements file yet — see Roadmap):
.venv/Scripts/pip install fastapi uvicorn anthropic chromadb sentence-transformers librosa numpy python-dotenv
cp .env.example .env          # leave USE_MOCK=true to run with no API key
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`. In **mock mode** the app runs with deterministic placeholder responses and no API calls. Live retrieval requires a vector index, which is **not included** (built from the private corpus); the pipeline, prompt skeletons, and endpoints can still be inspected, and the test suite runs against synthetic data:

```bash
.venv/Scripts/python -m pytest tests/unit -q   # synthetic signals, no private data, no network
```

## Example Workflow

A synthetic, rights-safe example lives in `sample_data/` (synthetic Callet-inspired notes + public-domain basics, example questions, and synthetic grounded-answer expectations for the evaluation harness) — demonstrating retrieval grounding and response checking without any copyrighted or private material.

## Repository Structure

```text
teaching-engine/
  app/                 # FastAPI app: /chat + audio/health/session routes, core RAG + citations
  services/
    audio/             # DSP feature-extraction skeletons (scoring rubrics withheld)
    rag/               # representative retrieval / embedding / vector-store wrappers + a librosa analyzer
    lesson/            # excluded from the snapshot (interface-level note only)
  teachers/jerome-callet/prompts/   # prompt-assembly skeletons (production content withheld)
  tests/               # unit tests + LLM-as-judge evaluation harness (synthetic data)
docs/                  # architecture + portfolio-snapshot notes
sample_data/           # synthetic / rights-cleared examples only
```

## Evaluation

An **LLM-as-judge** harness (`teaching-engine/tests/evaluation`) scores grounding, domain alignment, completeness, and helpfulness against expected points. The public repo ships synthetic evaluation examples; the original golden set (derived from copyrighted books) is excluded.

## Portfolio Snapshot

This repo is a deliberately reduced snapshot. [`docs/portfolio-snapshot.md`](docs/portfolio-snapshot.md) lists exactly what is included, what is omitted, and why — so it demonstrates engineering competence without exposing private/copyrighted material or being a runnable clone of the product.

## Rights and Privacy

This public repository excludes private recordings, private lesson notes, copyrighted instructional materials, proprietary indexes, and any real student data. Sample data is synthetic, public-domain, or rights-cleared. The git history has been scrubbed of the previously-committed private/copyrighted corpus.

## Pedagogical Focus

Jerome Callet-inspired trumpet pedagogy is used as a focused domain for AI-assisted teaching research. It is **not** an official Jerome Callet resource and does not claim to represent the full method authoritatively. Copyrighted and private instructional materials are excluded.

## License

**Source-available for portfolio review only — not open source.** All rights reserved. You may read the code to evaluate the author's work; you may not use, run, copy, modify, redistribute, or build on it without written permission. See [`LICENSE.md`](LICENSE.md).

## Limitations

A research/prototype system, not a substitute for an experienced trumpet teacher, a medical professional, or individualized diagnosis of playing mechanics. Any AI-generated feedback should be reviewed by a qualified teacher.

## Roadmap

- Grow the public synthetic / public-domain sample corpus and ship a small demo index
- Add a unified dependency manifest (`requirements.txt` / `pyproject`) and a one-command demo
- Add formal retrieval-evaluation metrics
