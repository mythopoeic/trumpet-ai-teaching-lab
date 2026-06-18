# Portfolio Snapshot — What's Included and What's Omitted

This public repository is a **portfolio snapshot**, not the full product. It is
curated to demonstrate engineering skills while deliberately **not** being a
runnable clone of the actual trumpet-teaching product, and while excluding all
private and copyrighted material.

## What is intentionally omitted (and why)

| Omitted | Why |
|---|---|
| Private source corpus (books, page scans, lesson recordings, transcripts, diarization, forum archive) | Copyrighted and/or personally identifying; never published. Purged from git history. |
| Built vector index (ChromaDB) | Derived from the private corpus. |
| Production system prompts (per-era, lesson, shared) | Encode the proprietary, Jerome Callet-inspired teaching behavior. Replaced with structural skeletons. |
| The stateful lesson/teaching product (curriculum, lesson-track composition, proficiency, drills, tracks, placement, selection, orchestration) | This is the product. Removed; a single interface-level note remains in `services/lesson/`. |
| Diagnostic rubrics & calibration (spit-buzz / tone / double-pedal scoring thresholds, calibration distributions) | Proprietary. Scorers/detectors reduced to documented skeletons; feature-extraction retained. |
| Ingestion / scraping / vision / audio-evaluation pipeline | Assumes private source materials; references private paths. Removed. |
| Internal tooling (autonomous-agent orchestration, ML training/dataset tooling, maintenance/query scripts), product roadmap/PRDs, internal agent docs | Not portfolio-relevant; reveal private workflow. Removed. |

## What is retained (engineering signal)

- FastAPI application skeleton (`teaching-engine/app`) with a grounded knowledge
  endpoint (`/chat`), audio endpoints, health, and session routes.
- Representative **RAG retrieval**: tiered character-budget retrieval over a
  vector store with sentence-transformer embeddings
  (`app/services/rag.py`, `rag_budget.py`, `services/rag/{vector_store,embedding_service,retriever}.py`).
- **Audio/DSP** feature-extraction skeletons (`services/audio/*_features.py`,
  `spectral_features.py`) with the scoring rubrics withheld.
- **Prompt-assembly architecture** (shared + per-era + retrieved-context layers)
  via skeleton prompts.
- **Evaluation harness** structure (`tests/evaluation`).
- Privacy/rights-aware design, `.env.example`, synthetic `sample_data/`, and
  architecture docs.

## Pedagogical focus

Jerome Callet-inspired trumpet pedagogy is referenced only at a high level as
the domain focus. This is a research/prototype snapshot, not an official or
authoritative resource, and not a substitute for a qualified teacher.
