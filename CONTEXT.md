# Context — domain glossary (portfolio snapshot)

Public, architecture-level terms for the retrieval/RAG and API layers of this
repository. Proprietary teaching concepts and product mechanics are excluded
from this snapshot (see `docs/portfolio-snapshot.md`).

## Language

**Era**:
A retrieval/routing dimension. The teaching focus spans more than one
historical approach, so retrieval and prompt selection can be scoped to an
"era". In this public snapshot, eras are a routing concept only; the
proprietary, era-specific teaching content is not included.

**Tier**:
The source-authority class of a retrieved chunk (e.g. book / media / forum).
Tier governs the retrieval budget and display ordering so one source type
cannot monopolize the context window.

**RAG (retrieval-augmented generation)**:
Answers are grounded in retrieved, curated source material rather than the
model's parametric knowledge. Retrieval uses a tiered character budget over a
vector index.

**Citation**:
A source attribution returned with an answer, carrying its tier and era, so
responses remain traceable to reviewed material.

**Knowledge mode**:
The grounded question-answering surface (`/chat`) retained in this snapshot.

> Note: the private source corpus, the built vector index, production prompts,
> and the stateful teaching ("lesson") product are intentionally excluded.
