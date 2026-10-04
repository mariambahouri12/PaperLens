# 🔍 PaperLens

**A production-quality multimodal RAG system for research papers.**

PaperLens ingests scientific PDFs, reconstructs their document structure, extracts text, equations, figures, images, and tables, and makes this information available to a retrieval-augmented generation (RAG) pipeline.

The extraction pipeline combines **MinerU** for document parsing with **Gemini API** for complex table extraction, while the downstream RAG pipeline remains fully local using **Qwen3 8B via Ollama**, hybrid retrieval, and local vector storage.

Built with **Clean Architecture** and **Clean Code** principles: external dependencies are isolated behind ports and adapters, making individual components replaceable without modifying the core application logic.

---

## ✨ Features

- 📄 **Structure-aware PDF extraction** — text, headings, sections, equations, figures, images, tables, captions, and reading order
- 🧮 **Equation extraction** — preserves scientific equations that were not reliably recovered by the previous extraction pipeline
- 🖼️ **First-class images** — extracted, stored on disk, and referenced by `image_id`
- 📊 **Complex table extraction** — MinerU detects table regions, while Gemini is used to reconstruct difficult tables that cannot be reliably represented from the initial extraction alone
- 🧩 **Hierarchical document structure** — headings are classified into sections and subsections using their numbering patterns
- 🔗 **Section-aware element association** — paragraphs, equations, images, tables, and other extracted elements are associated with the correct section/subsection
- 📑 **Automatic table of contents** — reconstructed from the detected heading hierarchy
- 🧠 **Hierarchical chunking** — chunks preserve their complete `section_path`
- 🔎 **Hybrid retrieval** — dense embeddings + BM25 fused with **Reciprocal Rank Fusion (RRF)**
- 🎯 **Configurable retrieval filtering** — minimum relevance score, maximum chunks, and maximum context tokens
- 🤖 **Local LLM generation** — Qwen3 8B via Ollama
- 🖼️ **Image retrieval** — resolves referenced `image_id`s to their actual files
- ⚙️ **Centralized configuration** — parameters are managed through `settings.py` / `.env`
- 📝 **Structured JSON logging**
- 🧪 **Unit and integration testing**

---

## 🔄 Why the Extraction Pipeline Was Reworked

The first version of PaperLens relied on a combination of **PyMuPDF (fitz)** and **pdfplumber** for PDF extraction.

Although this approach worked well for basic text and simple document elements, it became unreliable when processing scientific research papers with more complex layouts.

### Limitations of the previous approach

The previous extraction pipeline had two major weaknesses:

#### 1. Equations were not extracted reliably

Scientific papers contain mathematical notation, inline equations, displayed equations, and multi-line formulas.

The previous PyMuPDF-based extraction did not provide sufficiently reliable equation extraction and reconstruction for these documents.

As a result, important scientific information could be lost before reaching the RAG pipeline.

#### 2. Complex tables were difficult to reconstruct

`pdfplumber` worked reasonably well for simple tables, but scientific papers frequently contain:

- multi-level headers
- merged cells
- irregular column structures
- nested information
- tables spanning multiple rows or pages
- scientific notation
- visually structured tables without explicit PDF cell boundaries

For these cases, extracting text from the PDF was not enough to recover the original table structure accurately.

### New extraction strategy

PaperLens therefore moved to a **specialized multimodal extraction pipeline based on MinerU**.

```text
                    Scientific PDF
                          │
                          ▼
                ┌───────────────────┐
                │      MinerU        │
                │                   │
                │ Text              │
                │ Equations         │
                │ Images            │
                │ Figures            │
                │ Tables             │
                │ Reading order      │
                │ Document layout    │
                └─────────┬─────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
       Standard elements      Table regions
                │                   │
                │                   ▼
                │             Gemini API
                │                   │
                │          Complex table
                │           reconstruction
                │                   │
                └─────────┬─────────┘
                          ▼
                Hierarchical document
                          │
                          ▼
                Section / subsection
                   association
                          │
                          ▼
                Chunking + indexing
                          │
                          ▼
                  Hybrid Retrieval
                          │
                          ▼
                    Qwen3 8B
```

The goal is not to replace one model with another for the entire pipeline. Instead, each tool is used where it is most useful:

- **MinerU** handles general multimodal document extraction and layout reconstruction.
- **Gemini API** is used specifically to improve difficult table extraction.
- **Qwen3 8B via Ollama** remains the local LLM used for RAG answer generation.

This separation keeps the extraction pipeline specialized while keeping the main RAG system local.

---

## 🧩 Hierarchical Structure Reconstruction

Scientific papers use different heading conventions, for example:

```text
I. Introduction
II. Related Work
III. Methodology

A. Dataset
B. Experimental Setup
C. Evaluation

1. Data Collection
2. Model Training
3. Results

3.1 Dataset
3.2 Training
3.2.1 Hyperparameters
```

MinerU provides structural information about extracted text, including heading candidates. PaperLens then applies an additional **logical hierarchy detection layer** to determine the actual section level.

The hierarchy is inferred from the heading numbering pattern:

| Heading pattern         | Level |
| ----------------------- | ----: |
| `I. Introduction`       |     1 |
| `II. Methodology`       |     1 |
| `A. Dataset`            |     2 |
| `B. Experimental Setup` |     2 |
| `C. Evaluation`         |     2 |
| `1. Data Collection`    |     1 |
| `2. Model Training`     |     1 |
| `1.1 Dataset`           |     2 |
| `1.2 Training`          |     2 |
| `1.2.1 Hyperparameters` |     3 |

A specific ambiguity is handled explicitly: single-letter headings such as `C.` can also technically be interpreted as Roman numerals. Therefore, uppercase letter headings are checked before Roman numeral headings.

This produces a consistent hierarchical representation such as:

```text
Methodology
├── Dataset
├── Experimental Setup
└── Evaluation
```

Each extracted element receives the appropriate section context.

For example:

```json
{
  "type": "equation",
  "section": "Methodology",
  "subsection": "Model Architecture",
  "section_path": ["Methodology", "Model Architecture"]
}
```

The same mechanism is applied to:

- paragraphs
- equations
- images
- figures
- tables
- captions
- other extracted blocks

This is particularly important for downstream **chunking and RAG retrieval**, because retrieved information remains associated with the part of the paper where it originally appeared.

---

## 📊 Table Extraction Strategy

Tables are treated as first-class document elements.

The pipeline uses a two-stage strategy:

```text
PDF
 │
 ▼
MinerU
 │
 ├── detects table
 ├── extracts table region
 └── produces table representation
          │
          ▼
      Gemini API
          │
          ├── reconstruct columns
          ├── reconstruct rows
          ├── preserve headers
          └── recover complex structure
          │
          ▼
    Structured table
          │
          ▼
    Markdown rendering
          │
          ▼
       RAG chunk
```

Gemini is not used for every document element. It is specifically used when table reconstruction requires additional multimodal reasoning.

A cached result is stored for each processed table, avoiding unnecessary repeated API calls.

If Gemini extraction fails, PaperLens can fall back to the table representation generated by MinerU.

This provides a more robust extraction path while preserving a deterministic fallback.

---

## 🏗️ Architecture

PaperLens follows **Clean Architecture**: domain logic knows nothing about external libraries, the application layer orchestrates use cases, and infrastructure contains concrete implementations.

```text
┌─────────────────────────────────────────────────────┐
│                    INTERFACES                        │
│                  CLI commands                        │
│              ingest · ask · show-image              │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                   APPLICATION                        │
│                                                     │
│  use_cases: ingest_documents · answer_query         │
│  services: rrf_fusion · retrieval_filter            │
│            context_builder                           │
└─────────────────────────┬───────────────────────────┘
                          │
                          │ ports only
                          ▼
┌─────────────────────────────────────────────────────┐
│                     DOMAIN                           │
│                                                     │
│  entities · value_objects · repositories             │
│              pure business rules                    │
└─────────────────────────▲───────────────────────────┘
                          │ implements
┌─────────────────────────┴───────────────────────────┐
│                 INFRASTRUCTURE                      │
│                                                     │
│  extraction (MinerU · Gemini)                       │
│  chunking · embeddings · vector_store               │
│  BM25 · image_store · LLM · logging                 │
└─────────────────────────────────────────────────────┘
```

Every external dependency is isolated behind a port and implemented by an infrastructure adapter.

---

## 🔄 Pipeline

```text
Research Papers (data/)
        │
        ▼
Document Ingestion
        │
        ▼
Multimodal PDF Extraction
        │
        ├── MinerU
        │     ├── Text
        │     ├── Equations
        │     ├── Images
        │     ├── Figures
        │     ├── Captions
        │     └── Table regions
        │
        └── Gemini API
              └── Complex table reconstruction
        │
        ▼
Hierarchical Structure Reconstruction
        │
        ├── Heading detection
        ├── Section levels
        ├── Subsection levels
        └── Section paths
        │
        ▼
Normalized Document Elements
        │
        ├── paragraphs
        ├── equations
        ├── images
        ├── figures
        └── structured tables
        │
        ▼
Hierarchical Chunking
        │
        ▼
Chunks + Section / Image / Table Metadata
        │
        ▼
Indexing
   ├── Dense vectors → Qdrant local
   └── BM25 → rank_bm25
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
Retrieval Filtering
   ├── minimum relevance
   ├── maximum chunks
   └── context token budget
        │
        ▼
Local LLM
Qwen3 8B via Ollama
        │
        ▼
Answer + optional images / tables
```

---

## 📁 Project Structure

```text
paperlens/
│
├── main.py
├── requirements.txt
├── README.md
├── .env.example
│
├── app/
│   ├── domain/
│   │   ├── entities/
│   │   │   └── Document, Chunk, Image, Table, Query, Answer
│   │   ├── value_objects/
│   │   │   └── typed IDs, SectionPath
│   │   ├── repositories/
│   │   │   └── abstract ports
│   │   └── exceptions.py
│   │
│   ├── application/
│   │   ├── use_cases/
│   │   │   ├── ingest_documents.py
│   │   │   ├── answer_query.py
│   │   │   └── retrieve_images.py
│   │   └── services/
│   │       ├── rrf_fusion.py
│   │       ├── retrieval_filter.py
│   │       └── context_builder.py
│   │
│   ├── infrastructure/
│   │   ├── extraction/
│   │   │   └── mineru/
│   │   │       ├── extractor.py
│   │   │       ├── runner.py
│   │   │       ├── hierarchy.py
│   │   │       ├── table_parser.py
│   │   │       ├── table_enricher.py
│   │   │       ├── document_builder.py
│   │   │       ├── markdown_renderer.py
│   │   │       └── statistics.py
│   │   ├── chunking/
│   │   ├── embeddings/
│   │   ├── vector_store/
│   │   ├── bm25/
│   │   ├── image_store/
│   │   ├── llm/
│   │   └── logging/
│   │
│   ├── interfaces/
│   │   └── cli/
│   │       └── commands.py
│   │
│   └── config/
│       └── settings.py
│
├── data/
├── images/
│   └── <document_id>/
│       └── <image_id>.png
│
├── storage/
│   ├── vector/
│   └── bm25/
│
└── tests/
    ├── unit/
    └── integration/
```

---

## 🛠️ Technology Stack

| Component                | Technology                                         |
| ------------------------ | -------------------------------------------------- |
| Language                 | Python 3.11+                                       |
| PDF extraction           | **MinerU**                                         |
| Complex table extraction | **Gemini API**                                     |
| Embeddings               | `BAAI/bge-small-en-v1.5` via sentence-transformers |
| Vector store             | Qdrant local, on-disk mode                         |
| Lexical retrieval        | BM25 via `rank_bm25`                               |
| Fusion                   | Reciprocal Rank Fusion                             |
| LLM                      | Qwen3 8B via **Ollama**                            |
| Tokenizer                | `tiktoken` (`cl100k_base`)                         |
| Configuration            | `pydantic-settings`                                |
| CLI                      | Typer + Rich                                       |
| Testing                  | pytest                                             |

### API usage

Gemini is used only for the **specialized complex-table extraction stage**.

The main RAG generation model remains local:

```text
                    PaperLens
                        │
          ┌─────────────┴─────────────┐
          │                           │
     Extraction                  RAG Generation
          │                           │
     ┌────┴────┐                 ┌────┴────┐
     │         │                 │         │
   MinerU   Gemini API         Qdrant   Qwen3 8B
     │         │                BM25     Ollama
     └────┬────┘                 │         │
          │                      └────┬────┘
          ▼                           ▼
     Structured                  Final Answer
     Document
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.11+**
- **MinerU** installed and configured
- **Ollama** installed locally
- A **Gemini API key** for complex table extraction
- Recommended OS: **Linux / macOS / WSL2**
- Native Windows is also supported depending on the MinerU environment

Pull the local generation model:

```bash
ollama pull qwen3:8b
```

Configure the Gemini API key in `.env` according to the project's configuration.

### 2. Install

```bash
git clone <repository-url>
cd paperlens

python -m venv .venv

# Linux / macOS / WSL2
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
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

The ingestion pipeline performs:

```text
PDF
 ↓
MinerU extraction
 ↓
Complex table enrichment with Gemini
 ↓
Hierarchical section reconstruction
 ↓
Document normalization
 ↓
Hierarchical chunking
 ↓
Embedding + BM25 indexing
```

Images are written to:

```text
images/<document_id>/<image_id>.png
```

Vectors are stored in:

```text
storage/vector/
```

and BM25 indices in:

```text
storage/bm25/
```

### 5. Ask questions

```bash
# Text question
python main.py ask "What methodology does this paper use?"

# Request a specific figure
python main.py ask "Show me the diagram of the proposed architecture." --image-only

# Explanation + figure
python main.py ask "Explain the architecture and show me the figure." --with-images
```

---

## 🖼️ Image & Table Handling

Images and tables are treated as **first-class document elements**.

### Images

Every extracted image receives a unique `image_id`:

```text
images/
└── <document_id>/
    ├── <image_id>.png
    ├── <image_id>.png
    └── ...
```

Each image keeps:

- `image_id`
- caption
- page number
- section path
- source document

Chunks reference images through their IDs rather than embedding binary image data into the chunk text.

Example:

```json
{
  "chunk_id": "doc_ab12_p7_003",
  "text": "...text surrounding the figure...",
  "metadata": {
    "section_path": ["Methodology", "Model Architecture"],
    "image_ids": ["doc_ab12_img_007_000"],
    "image_captions": ["Figure 3: Architecture of the proposed system"],
    "page_numbers": [7]
  }
}
```

### Tables

Tables preserve their row/column structure and are rendered as Markdown for downstream RAG processing.

Example:

```markdown
| Model   | Accuracy |   F1 |
| ------- | -------: | ---: |
| Model A |    92.1% | 0.89 |
| Model B |    94.3% | 0.92 |
```

The structured representation allows the LLM to reason about relationships between rows and columns rather than receiving a flattened sequence of text.

---

## 🧠 Section-Aware RAG

PaperLens does not treat a paper as an unordered collection of text chunks.

Each element is associated with its hierarchical location:

```text
Paper
│
├── Introduction
│
├── Related Work
│
├── Methodology
│   ├── Dataset
│   ├── Model Architecture
│   │   └── Training Procedure
│   └── Experimental Setup
│
└── Results
    ├── Quantitative Results
    └── Discussion
```

An extracted element can therefore carry:

```json
{
  "section_path": ["Methodology", "Model Architecture", "Training Procedure"]
}
```

This information is preserved during chunking and retrieval.

As a result, retrieved content remains traceable to the original part of the paper.

---

## ⚙️ Configuration

All tunable parameters are centralized in:

```text
app/config/settings.py
```

and can be overridden through `.env`.

Typical configuration includes:

| Key                             | Default                  | Purpose                     |
| ------------------------------- | ------------------------ | --------------------------- |
| `PAPERLENS_DATA_DIR`            | `./data`                 | PDF input directory         |
| `PAPERLENS_IMAGE_DIR`           | `./images`               | Extracted images            |
| `PAPERLENS_STORAGE_DIR`         | `./storage`              | Persistent indices          |
| `PAPERLENS_EMBEDDING_MODEL`     | `BAAI/bge-small-en-v1.5` | Dense embedding model       |
| `PAPERLENS_LLM_MODEL`           | `qwen3:8b`               | Local generation model      |
| `PAPERLENS_SECTION_MAX_TOKENS`  | `1200`                   | Section splitting threshold |
| `PAPERLENS_CHUNK_SIZE`          | `500`                    | Fallback chunk size         |
| `PAPERLENS_CHUNK_OVERLAP`       | `80`                     | Fallback overlap            |
| `PAPERLENS_MIN_RELEVANCE_SCORE` | `0.005`                  | RRF score threshold         |
| `PAPERLENS_MAX_CHUNKS`          | `10`                     | Maximum returned chunks     |
| `PAPERLENS_MAX_CONTEXT_TOKENS`  | `6000`                   | LLM context budget          |
| `PAPERLENS_RRF_K`               | `60`                     | RRF smoothing constant      |
| `PAPERLENS_TEMPERATURE`         | `0`                      | LLM temperature             |
| `PAPERLENS_NUM_CTX`             | `8384`                   | LLM context window          |
| `PAPERLENS_LOG_LEVEL`           | `INFO`                   | Log verbosity               |

Gemini-specific settings are also kept in the extraction configuration.

---

## 🧪 Tests

```bash
pytest -q
```

### Unit tests

```text
tests/unit/
├── test_section_tree.py
├── test_chunker.py
├── test_rrf.py
├── test_retrieval_filter.py
└── test_ids.py
```

These cover:

- heading detection
- hierarchical section paths
- section-aware chunking
- fallback chunking
- overlap
- RRF fusion
- retrieval filtering
- typed IDs

### Integration tests

```text
tests/integration/
├── test_extraction.py
├── test_image_store.py
└── test_end_to_end.py
```

Extraction tests cover the multimodal extraction pipeline and normalized document elements.

---

## 🧭 Design Principles

### Clean Architecture

- `domain/` contains pure business rules.
- `application/` depends only on the domain.
- `infrastructure/` contains concrete adapters.
- `main.py` is the composition root.

### Ports & Adapters

External dependencies are isolated behind interfaces:

| Port                    | Adapter                       |
| ----------------------- | ----------------------------- |
| `DocumentExtractorPort` | `MinerUExtractor`             |
| `ChunkerPort`           | `HierarchicalChunker`         |
| `EmbedderPort`          | `SentenceTransformerEmbedder` |
| `VectorStorePort`       | `QdrantLocalStore`            |
| `BM25IndexPort`         | `RankBM25Index`               |
| `ImageStorePort`        | `FilesystemImageStore`        |
| `LLMPort`               | `OllamaLLM`                   |

The extraction adapter itself encapsulates the MinerU execution and Gemini-based table enrichment.

### Extensibility

The extraction layer is intentionally separated from the rest of the RAG pipeline.

A future extractor can replace MinerU without changing:

- chunking
- embeddings
- retrieval
- RRF fusion
- context construction
- LLM generation

Likewise, the table enrichment strategy can evolve independently.

### Error Handling

A failure affecting one document or table should not terminate the entire ingestion process.

Failures are logged with relevant context such as:

```text
document_id
table_id
image_id
page_number
section_path
```

For table enrichment, MinerU output can be retained as a fallback when Gemini processing fails.

---

## 🎯 Example Session

```bash
$ python main.py ingest

{
  "discovered": 5,
  "ingested": 5,
  "failed": 0,
  "chunks": 412,
  "images": 87,
  "tables": 24
}
```

Ask a text question:

```bash
$ python main.py ask "What methodology does this paper use?"
```

Example answer:

```text
The paper uses a Bayesian hierarchical model combined
with variational inference...

Relevant section:
Methodology → Model Architecture
```

Retrieve a figure:

```bash
$ python main.py ask "Show me the architecture diagram." --image-only
```

Example:

```text
doc_ab12_img_007_000
Figure 3: Architecture of the proposed system

→ images/doc_ab12/doc_ab12_img_007_000.png
```

Retrieve information from a complex table:

```bash
$ python main.py ask "What is the best F1-score reported in Table 4?"
```

PaperLens can retrieve the structured table representation together with its section context and provide the answer.

---

## 🔮 Roadmap

- [ ] Persistent near-duplicate detection across runs
- [ ] Optional multimodal image embeddings (CLIP / SigLIP)
- [ ] Additional extractors (LaTeX / arXiv, HTML, DOCX)
- [ ] Parallel ingestion
- [ ] Persistent chunk cache keyed by content hash
- [ ] Prometheus / StatsD metrics export
- [ ] Integration with external vector databases
- [ ] Improved table validation and structural consistency checks
- [ ] Automatic extraction quality evaluation
- [ ] More robust multimodal retrieval

---

## 📜 License

MIT.

---

> **PaperLens** — turn research papers into structured, traceable, and retrieval-ready knowledge for RAG.
