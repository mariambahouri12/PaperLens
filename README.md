# 🔍 PaperLens

**A production-quality multimodal RAG system for research papers.**

PaperLens ingests PDFs of scientific papers, recovers their structure
(sections, figures, tables, captions), and lets you ask questions about
them — including requests for specific diagrams or tables — using a
fully local, open-source stack.

Built with **Clean Architecture** and **Clean Code** principles: every
external dependency (PDF parser, embedding model, vector store, BM25,
LLM, image store) sits behind a port and can be swapped with a single
adapter class.

---

## ✨ Features

- 📄 **Structure-aware PDF extraction** — metadata, headings, sections, paragraphs, figures, tables, captions, reading order, bounding boxes
- 🖼️ **First-class images** — extracted, stored on disk, and referenced everywhere by `image_id` (never duplicated into chunk text)
- 📊 **Structured tables** — preserved as rows/columns, rendered as Markdown in chunks (never flattened into plain text)
- 🧩 **Hierarchical chunking** — section-aware chunks that keep the full `section_path`, with a fixed-size + overlap fallback for large sections
- 🔎 **Hybrid retrieval** — dense embeddings + BM25, fused with **Reciprocal Rank Fusion (RRF)**
- 🎯 **Configurable filtering** — minimum relevance score, max chunks, max context tokens
- 🤖 **Local LLM generation** — Qwen3 8B via Ollama, no paid API, no external service
- 🖼️ **Image retrieval** — resolves referenced `image_id`s to actual files from the image store
- ⚙️ **Fully configurable** — every parameter lives in one place (`settings.py` / `.env`)
- 📝 **Structured JSON logging** across the whole pipeline
- 🧪 **Tested** — unit tests for domain logic, integration tests for extraction and image store

---

## 🏗️ Architecture

PaperLens follows **Clean Architecture**: domain logic knows nothing about
libraries, the application layer orchestrates use cases, and infrastructure
holds every concrete adapter.

```
┌─────────────────────────────────────────────────────┐
│                  INTERFACES (CLI)                    │
│              ingest · ask · show-image               │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                   APPLICATION                        │
│   use_cases: ingest_documents · answer_query         │
│   services:  rrf_fusion · retrieval_filter ·         │
│              context_builder                         │
└─────────────────────────┬───────────────────────────┘
                          │ (ports only)
                          ▼
┌─────────────────────────────────────────────────────┐
│                     DOMAIN                           │
│   entities · value_objects · repositories (ports)    │
│   (no library import — pure business rules)          │
└─────────────────────────▲───────────────────────────┘
                          │ implements
┌─────────────────────────┴───────────────────────────┐
│                 INFRASTRUCTURE                       │
│   extraction (PyMuPDF · pdfplumber) · chunking ·     │
│   embeddings (sentence-transformers) · vector_store  │
│   (Qdrant local) · bm25 (rank_bm25) · image_store ·  │
│   llm (Ollama) · logging                             │
└─────────────────────────────────────────────────────┘
```

Every port (`DocumentExtractorPort`, `ChunkerPort`, `EmbedderPort`,
`VectorStorePort`, `BM25IndexPort`, `ImageStorePort`, `LLMPort`) is
declared in `app/domain/repositories/` and implemented once in
`app/infrastructure/`. Swapping any of them requires **one new adapter
class and one line in the composition root** — nothing else.

---

## 🔄 Pipeline

```
Research Papers (data/)
        │
        ▼
Document Ingestion
        │
        ▼
Structure-Aware Extraction         ← PyMuPDF + pdfplumber
        │
        ▼
Normalized Document Elements
(paragraphs · headings · images · tables · captions)
        │
        ▼
Hierarchical Chunking               ← section-aware + fixed-size fallback
        │
        ▼
Chunks + Image / Table Metadata
        │
        ▼
Indexing
  ├── Dense vectors (Qdrant local)
  └── BM25 (rank_bm25)
        │
        ▼
Hybrid Retrieval
  ├── Semantic search
  └── BM25
        │
        ▼
RRF Fusion
        │
        ▼
Retrieval Filtering                 ← min score · max chunks · max tokens
        │
        ▼
Local LLM (Qwen3 8B via Ollama)
        │
        ▼
Answer + optional images / tables
```

---

## 📁 Project Structure

```
paperlens/
│
├── main.py                                    # composition root (CLI entrypoint)
├── requirements.txt
├── README.md
├── .env.example
│
├── app/
│   ├── domain/                                # pure business rules
│   │   ├── entities/                          # Document, Chunk, Image, Table, Query, Answer
│   │   ├── value_objects/                     # typed IDs, SectionPath
│   │   ├── repositories/                      # abstract ports
│   │   └── exceptions.py
│   │
│   ├── application/                           # use cases + services
│   │   ├── use_cases/
│   │   │   ├── ingest_documents.py
│   │   │   ├── answer_query.py
│   │   │   └── retrieve_images.py
│   │   └── services/
│   │       ├── rrf_fusion.py
│   │       ├── retrieval_filter.py
│   │       └── context_builder.py
│   │
│   ├── infrastructure/                        # concrete adapters
│   │   ├── extraction/
│   │   ├── chunking/
│   │   ├── embeddings/
│   │   ├── vector_store/
│   │   ├── bm25/
│   │   ├── image_store/
│   │   ├── llm/
│   │   └── logging/
│   │
│   ├── interfaces/cli/                        # user-facing commands
│   │   └── commands.py
│   │
│   └── config/settings.py                     # single source of truth
│
├── data/                                      # input PDFs
├── images/                                    # extracted images
│   └── <document_id>/<image_id>.png
├── storage/                                   # persistent indices
│   ├── vector/
│   └── bm25/
└── tests/
    ├── unit/
    └── integration/
```

---

## 🛠️ Technology Stack

| Component         | Technology                                       |
| ----------------- | ------------------------------------------------ |
| Language          | Python 3.11+                                     |
| PDF extraction    | PyMuPDF (fitz) + pdfplumber                      |
| Embeddings        | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Vector store      | Qdrant (local, on-disk mode)                     |
| Lexical retrieval | BM25 via `rank_bm25`                             |
| Fusion            | Reciprocal Rank Fusion (custom)                  |
| LLM               | Qwen3 8B via **Ollama** (local)                  |
| Tokenizer         | tiktoken (`cl100k_base`)                         |
| Config            | pydantic-settings                                |
| CLI               | Typer + Rich                                     |
| Testing           | pytest                                           |

**Everything is free and open-source. No paid API. No cloud service.**

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed locally
- Recommended OS: **Linux / macOS / WSL2** (works on native Windows too)

Pull the model:

```bash
ollama pull qwen3:8b
```

### 2. Install

```bash
git clone <repository-url>
cd paperlens

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env                # edit if needed
```

### 3. Add papers

```bash
mkdir -p data
cp ~/papers/*.pdf data/
```

### 4. Ingest

```bash
python main.py ingest
```

This runs the full pipeline: extract → chunk → embed → index. Images
are written to `images/<document_id>/<image_id>.png`, vectors to
`storage/vector/`, BM25 to `storage/bm25/`.

### 5. Ask

```bash
# Text-only question
python main.py ask "What methodology does this paper use?"

# Request a specific figure
python main.py ask "Show me the diagram of the proposed architecture." --image-only

# Explanation + figure
python main.py ask "Explain the architecture and show me the figure." --with-images
```

---

## ⚙️ Configuration

All parameters live in `app/config/settings.py` and are overridable via
`.env` (prefix `PAPERLENS_`).

| Key                             | Default                  | Purpose                           |
| ------------------------------- | ------------------------ | --------------------------------- |
| `PAPERLENS_DATA_DIR`            | `./data`                 | Where PDFs are read from          |
| `PAPERLENS_IMAGE_DIR`           | `./images`               | Where extracted images are stored |
| `PAPERLENS_STORAGE_DIR`         | `./storage`              | Where indices are persisted       |
| `PAPERLENS_EMBEDDING_MODEL`     | `BAAI/bge-small-en-v1.5` | Dense embedding model             |
| `PAPERLENS_LLM_MODEL`           | `qwen3:8b`               | Local generation model            |
| `PAPERLENS_SECTION_MAX_TOKENS`  | `1200`                   | Above this, sections are split    |
| `PAPERLENS_CHUNK_SIZE`          | `500`                    | Fallback chunk size               |
| `PAPERLENS_CHUNK_OVERLAP`       | `80`                     | Fallback overlap                  |
| `PAPERLENS_MIN_RELEVANCE_SCORE` | `0.005`                  | RRF score threshold               |
| `PAPERLENS_MAX_CHUNKS`          | `10`                     | Max returned chunks               |
| `PAPERLENS_MAX_CONTEXT_TOKENS`  | `6000`                   | LLM context budget                |
| `PAPERLENS_RRF_K`               | `60`                     | RRF smoothing constant            |
| `PAPERLENS_TEMPERATURE`         | `0`                      | LLM temperature                   |
| `PAPERLENS_NUM_CTX`             | `8384`                   | LLM context window                |
| `PAPERLENS_LOG_LEVEL`           | `INFO`                   | Log verbosity                     |

No magic numbers anywhere in the codebase. **Every** tunable lives here.

---

## 🧪 Tests

```bash
pytest -q
```

- **`tests/unit/`** — pure domain and application logic (no I/O)
  - `test_section_tree.py` — heading detection, hierarchical paths
  - `test_chunker.py` — section-aware chunking, fallback, overlap
  - `test_rrf.py` — RRF fusion math
  - `test_retrieval_filter.py` — thresholds, max chunks, token budget
  - `test_ids.py` — value objects

- **`tests/integration/`** — infrastructure against real artifacts
  - `test_extraction.py` — PDF parsing, images, captions (builds PDFs on the fly, no binary fixture)
  - `test_image_store.py` — filesystem round-trip
  - `test_end_to_end.py` — full ingest + query (skipped if no PDF in `data/`)

Tests run fast (< 1 s each) and never require Ollama or an embedding
model unless explicitly exercised.

---

## 🧭 Design Principles

### Clean Architecture

- `domain/` has **zero** library imports.
- `application/` depends only on `domain/` — never on `infrastructure/`.
- `infrastructure/` implements domain ports; each adapter is small and focused.
- `main.py` is the **only** composition root.

### Ports & Adapters

Every external dependency is hidden behind an interface:

| Port                    | Adapter(s) shipped                             |
| ----------------------- | ---------------------------------------------- |
| `DocumentExtractorPort` | `PyMuPDFExtractor` (+ `pdfplumber` for tables) |
| `ChunkerPort`           | `HierarchicalChunker`                          |
| `EmbedderPort`          | `SentenceTransformerEmbedder`                  |
| `VectorStorePort`       | `QdrantLocalStore`                             |
| `BM25IndexPort`         | `RankBM25Index`                                |
| `ImageStorePort`        | `FilesystemImageStore`                         |
| `LLMPort`               | `OllamaLLM`                                    |

### Extensibility

Adding a new format, a new embedding model, a new vector DB, or a new
LLM is a **one-adapter, one-line** change. No core pipeline code is
touched.

### Error handling

One bad document never kills a batch. Failures are logged with context
(`document_id`, `chunk_id`, `image_id`, `page_number`) and the pipeline
continues.

---

## 🖼️ Image & Table Handling

Images and tables are **first-class citizens**, not flattened into text.

- Every image gets a unique `image_id` and is written to
  `images/<document_id>/<image_id>.png`.
- Every image's **caption**, **section path** and **page** are stored in
  metadata.
- A chunk that references an image stores only the `image_id` and its
  caption — **never** the binary content.
- At query time, if the top chunks reference images, PaperLens resolves
  each `image_id` back to the actual file.

Example chunk metadata:

```json
{
  "chunk_id": "doc_ab12_pResults_s0",
  "text": "...text surrounding the figure...",
  "metadata": {
    "section_path": ["Bayesian Methods", "Results"],
    "image_ids": ["doc_ab12_img_007_000"],
    "image_captions": ["Figure 3: Architecture of the proposed system"],
    "page_numbers": [7]
  }
}
```

Tables are stored the same way, with their full row/column structure
preserved. The chunk text contains a Markdown rendering of the table so
the LLM sees its structure.

---

## 🎯 Example Session

```bash
$ python main.py ingest
{"discovered": 5, "ingested": 5, "failed": 0, "chunks": 412, "images": 87}

$ python main.py ask "What methodology does this paper use?"

Answer:
The paper uses a Bayesian hierarchical model combined with variational
inference (see chunk doc_ab12_pMethods_full) ...

$ python main.py ask "Show me the architecture diagram." --image-only
doc_ab12_img_007_000   Figure 3: Architecture of the proposed system   -> images/doc_ab12/doc_ab12_img_007_000.png

$ python main.py ask "Explain the architecture and show me the figure." --with-images

Answer:
The proposed system is a three-stage pipeline: (1) input encoding ...
[full explanation]

Referenced images:
  - doc_ab12_img_007_000: Figure 3: Architecture of the proposed system
    images/doc_ab12/doc_ab12_img_007_000.png
```

---

## 🔮 Roadmap

- [ ] Persistent near-duplicate detection across runs
- [ ] Optional multimodal image embeddings (CLIP / SigLIP)
- [ ] Additional extractors (LaTeX / arXiv, HTML, DOCX)
- [ ] Parallel ingestion (process pool)
- [ ] Persistent chunk cache keyed by content hash
- [ ] Prometheus / StatsD metrics export
- [ ] Integration with external vector databases (Qdrant server, Weaviate)

---

## 📜 License

MIT.

---

> **PaperLens** — turn research papers into structured, traceable, and retrieval-ready knowledge for RAG.
