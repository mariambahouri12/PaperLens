# 🔍 PaperLens

**A structure-aware multimodal RAG system for scientific research papers.**

PaperLens transforms scientific PDFs into structured, searchable knowledge. It combines multimodal document extraction, hierarchical document reconstruction, structure-aware chunking, hybrid retrieval, and local LLM generation to answer questions grounded in research papers.

The extraction pipeline uses **MinerU** for document parsing and the **Gemini API** for specialized complex-table reconstruction. The downstream RAG pipeline uses local embeddings, Qdrant, BM25, Reciprocal Rank Fusion (RRF), and **Qwen3 8B through Ollama**.

Built around **Clean Architecture** and **Ports & Adapters**, PaperLens separates domain logic from external tools so that infrastructure components can be replaced without rewriting the core application.

---

## ✨ Features

- 📄 **Structure-aware PDF extraction** — extracts text, headings, equations, figures, images, tables, captions, and reading order.
- 🧮 **Scientific equation preservation** — retains equations as structured elements associated with their document context.
- 🖼️ **Image and figure handling** — stores extracted images on disk and associates them with metadata such as image IDs, captions, pages, and sections.
- 📊 **Complex table reconstruction** — combines MinerU's table extraction with Gemini-assisted reconstruction when needed, with a fallback to the MinerU representation.
- 🧩 **Hierarchical structure reconstruction** — detects sections and subsections from heading patterns and numbering conventions.
- 🧾 **Front-matter detection** — separates authors and abstracts into dedicated sections, including cases where the abstract heading is not detected correctly by the extractor.
- 🔗 **Section-aware element association** — associates paragraphs, equations, lists, images, tables, and captions with their corresponding document sections.
- 📑 **Automatic table-of-contents reconstruction** — builds a hierarchy from detected headings.
- ✂️ **Hierarchical chunking** — respects section boundaries, token budgets, overlap rules, reading order, and content types.
- 🔎 **Hybrid retrieval** — combines dense semantic search and BM25 lexical retrieval through Reciprocal Rank Fusion.
- 🎯 **Retrieval filtering** — supports configurable relevance thresholds, maximum chunk counts, and context token budgets.
- 🤖 **Local answer generation** — uses Qwen3 8B through Ollama for question answering.
- 📚 **Source-aware answers** — displays concise source references using retrieved chunk metadata, including document, page, and section.
- 🖥️ **Streamlit interface** — provides an interactive research workspace for asking questions, viewing answers, inspecting sources, and displaying retrieved figures.
- ⚙️ **Centralized configuration** — manages settings through `app/config/settings.py` and environment variables.
- 📝 **Structured logging** — supports structured application logs for diagnostics.
- 🧪 **Automated testing** — supports unit and integration tests through pytest.

---

## 🏛️ Architecture at a glance

PaperLens separates document extraction, domain modeling, indexing, retrieval, and answer generation into distinct stages.

```text
                 Scientific PDFs
                        │
                        ▼
               Multimodal Extraction
                 MinerU + Gemini
                        │
                        ▼
             Document Structure Recovery
          Sections · Equations · Tables · Images
                        │
                        ▼
                 Domain Document
                        │
                        ▼
               Hierarchical Chunking
                        │
                        ▼
                  Embeddings
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        Qdrant Local              BM25
       Dense Retrieval      Lexical Retrieval
             │                     │
             └──────────┬──────────┘
                        ▼
                    RRF Fusion
                        │
                        ▼
                Retrieval Filtering
                        │
                        ▼
                 Context Assembly
                        │
                        ▼
                 Qwen3 8B / Ollama
                        │
                        ▼
                 Answer + Sources
                        │
                        ▼
                 Streamlit Interface
```

The Gemini API is reserved for specialized table extraction. The main retrieval and answer-generation pipeline is designed to run locally.

---

## 🔄 Why the extraction pipeline was redesigned

The initial implementation relied on PyMuPDF (`fitz`) and `pdfplumber` for PDF extraction. While useful for ordinary text and simple layouts, these tools alone were not sufficiently reliable for the variety of scientific document structures PaperLens needed to preserve.

### 1. Scientific equations

Research papers contain inline mathematics, displayed equations, multi-line formulas, and scientific notation. Extraction must preserve these elements and their relationship to surrounding text so they remain useful during retrieval.

### 2. Complex tables

Scientific tables may contain multi-level headers, merged cells, irregular columns, nested information, and visual structures that cannot be recovered reliably by simply flattening PDF text.

### The current extraction strategy

PaperLens uses a specialized multimodal extraction workflow:

1. **MinerU** performs general document parsing and extracts structural elements.
2. **Gemini API** can reconstruct complex tables when the initial table representation needs improvement.
3. PaperLens normalizes the extraction output into a domain-level document model.
4. The resulting document is passed to the chunking and indexing pipeline.

```text
Scientific PDF
      │
      ▼
    MinerU
      │
      ├── Text and headings
      ├── Equations
      ├── Images and figures
      ├── Captions
      └── Table regions
               │
               ▼
       Gemini API (when needed)
               │
               ▼
       Table representation
               │
               ▼
       Document normalization
               │
               ▼
       Hierarchical chunking
```

This design assigns each tool a specific responsibility rather than using one model for every stage.

---

## 🧩 Hierarchical structure reconstruction

Scientific papers use different heading conventions, including Roman numerals, uppercase letters, numbered sections, and decimal subsections.

For example:

```text
I. Introduction
II. Related Work
III. Methodology

A. Dataset
B. Experimental Setup
C. Evaluation

1. Data Collection
2. Model Training

3.1 Dataset
3.2 Training
3.2.1 Hyperparameters
```

MinerU provides heading candidates and structural information. PaperLens applies an additional hierarchy-detection layer to infer section levels from heading patterns.

| Heading pattern         | Intended level |
| ----------------------- | -------------: |
| `I. Introduction`       |              1 |
| `V. Discussion`         |              1 |
| `A. Dataset`            |              2 |
| `B. Experimental Setup` |              2 |
| `1. Data Collection`    |              1 |
| `3.1 Dataset`           |              2 |
| `3.2.1 Hyperparameters` |              3 |

Roman numerals and subsection letters require explicit handling because the same characters can be ambiguous. PaperLens applies dedicated rules to distinguish common heading patterns.

A reconstructed hierarchy might look like this:

```text
Methodology
├── Dataset
├── Experimental Setup
└── Evaluation
```

Each extracted element is associated with its section context. For example:

```json
{
  "type": "equation",
  "section_path": ["Methodology", "Model Architecture"]
}
```

The same principle applies to paragraphs, lists, tables, images, figures, equations, and captions.

### Front matter

PaperLens applies special rules to the beginning of a document to identify author information and the abstract. Abstract detection recognizes common heading forms such as `Abstract—`, `Abstract:`, and `ABSTRACT.` even when the extraction engine does not classify them as headings.

These rules are limited to the front matter to reduce the risk of misclassifying later occurrences of the word “Abstract.”

---

## 📊 Complex table extraction

Tables are treated as first-class document elements rather than undifferentiated text.

```text
PDF
 │
 ▼
MinerU table detection and extraction
 │
 ▼
Initial table representation
 │
 ▼
Is additional reconstruction needed?
 │
 ├── No ────────────────┐
 │                      │
 └── Yes                │
      │                 │
      ▼                 │
  Gemini API            │
      │                 │
      ▼                 │
  Reconstructed table   │
      │                 │
      └────────┬────────┘
               ▼
       Structured table
               │
               ▼
        Markdown rendering
               │
               ▼
          RAG chunk
```

The table-enrichment stage can use cached results to avoid unnecessary repeated API calls. If Gemini processing fails, the pipeline can retain the available MinerU representation as a fallback.

The objective is to improve difficult tables without requiring the external API for every document element.

---

## ✂️ Hierarchical chunking

PaperLens chunks the normalized domain `Document`, not the raw extraction JSON.

```text
Extraction output
       │
       ▼
DocumentMapper
       │
       ▼
Domain Document
       │
       ▼
HierarchicalChunker
       │
       ▼
List[Chunk]
```

### Chunking rules

| Content                 | Strategy                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| Consecutive text        | Merge within the same section while respecting the token budget                               |
| Token budget exceeded   | Start another chunk and carry over eligible trailing content according to the overlap setting |
| Oversized text unit     | Split into overlapping token windows                                                          |
| Different content types | Preserve content-type boundaries and reading order                                            |
| Equations               | Associate with surrounding explanatory text according to the chunking rules                   |
| Lists                   | Pack item by item to avoid cutting through individual items                                   |
| Tables                  | Keep each table in its own chunk, including available caption, table content, and footnotes   |
| Images and charts       | Create dedicated chunks based on available captions and metadata                              |
| Images without captions | Skip when no useful text is available for embedding                                           |
| Section boundaries      | Never merge chunks across different sections                                                  |

### Chunk metadata

Each chunk carries metadata that helps maintain traceability:

| Field          | Purpose                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `chunk_id`     | Unique chunk identifier                                                |
| `document_id`  | Identifier shared by chunks from the same document                     |
| `filename`     | Original PDF filename                                                  |
| `chunk_index`  | Position within the document                                           |
| `chunk_types`  | Types such as `text`, `equation`, `list`, `table`, `image`, or `chart` |
| `section_path` | Ordered hierarchy of section and subsection titles                     |
| `page_numbers` | Pages covered by the chunk                                             |
| `image_id`     | Identifier for an associated image, when available                     |
| `image_path`   | Filesystem location of an associated image, when available             |
| `token_count`  | Token count used by filtering and context-budget logic                 |

Example:

```json
{
  "chunk_id": "document_0012",
  "text": "Description of the proposed architecture...",
  "token_count": 116,
  "metadata": {
    "document_id": "document_id",
    "filename": "research_paper.pdf",
    "chunk_index": 12,
    "chunk_types": ["image"],
    "section_path": [
      "IV. Experimental Results",
      "B. Model Poisoning Attack Impact"
    ],
    "page_numbers": [3],
    "image_id": "image_1",
    "image_path": "images/document_id/image_1.jpg"
  }
}
```

The example illustrates the metadata schema; the identifiers and paths are illustrative.

---

## 🔎 Hybrid retrieval and source-aware answers

PaperLens combines two complementary retrieval methods.

### Dense retrieval

A sentence-transformer embedding model converts chunks and user queries into vector representations. Qdrant stores the vectors and supports semantic retrieval.

### Lexical retrieval

BM25 finds chunks that share meaningful terms with the query. This is useful for exact terminology, model names, abbreviations, and technical expressions.

### Reciprocal Rank Fusion

The dense and lexical result lists are combined using RRF:

```text
Dense results ──────┐
                    ├──► RRF Fusion
BM25 results ───────┘         │
                              ▼
                      Retrieval Filtering
                              │
                              ▼
                      Context Construction
                              │
                              ▼
                       Local LLM Answer
```

Filtering applies configured relevance thresholds, chunk limits, and context-token budgets before context construction.

### Source references

The application retains metadata for the chunks selected for the answer. The interface can display concise references in this form:

```text
Source:
research_paper.pdf — p. 4 — section 3.1
```

Source metadata is obtained from retrieved chunks rather than relying on the LLM to invent document names, pages, or section identifiers.

The Streamlit interface also supports displaying images associated with the answer when they are available.

---

## 🏗️ Clean Architecture

PaperLens separates core business logic from infrastructure-specific implementations.

```text
┌─────────────────────────────────────────────┐
│                 INTERFACES                  │
│                                             │
│             CLI · Streamlit                 │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                APPLICATION                  │
│                                             │
│  Use cases:                                 │
│    ingest_documents                         │
│    answer_query                             │
│    retrieve_images                          │
│                                             │
│  Services:                                  │
│    hybrid_retriever · rrf_fusion             │
│    retrieval_filter · context_builder       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                  DOMAIN                     │
│                                             │
│    Entities · Value objects · Ports         │
│              Business rules                 │
└──────────────────────▲──────────────────────┘
                       │ implements
┌──────────────────────┴──────────────────────┐
│               INFRASTRUCTURE                │
│                                             │
│ Extraction · Chunking · Embeddings          │
│ Qdrant · BM25 · Image storage · Ollama      │
│ Logging · Checkpoint persistence            │
└─────────────────────────────────────────────┘
```

### Design principles

- **Domain independence:** domain entities and rules avoid dependencies on concrete external services.
- **Application orchestration:** use cases coordinate retrieval, indexing, context construction, and answer generation.
- **Ports and adapters:** infrastructure implementations satisfy application/domain interfaces.
- **Dependency isolation:** external libraries are concentrated in infrastructure and composition-root code.
- **Replaceability:** extraction, embeddings, vector storage, and LLM adapters can evolve independently when their contracts remain stable.

---

## 📁 Project structure

The following is a representative view of the main application modules:

```text
paperlens/
├── main.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── app/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── answer.py
│   │   │   ├── block.py
│   │   │   ├── chunk.py
│   │   │   ├── document.py
│   │   │   ├── image.py
│   │   │   ├── query.py
│   │   │   └── section.py
│   │   ├── repositories/
│   │   ├── value_objects/
│   │   └── exceptions.py
│   │
│   ├── application/
│   │   ├── mappers/
│   │   │   └── chunk_payload_mapper.py
│   │   ├── services/
│   │   │   ├── context_builder.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── retrieval_filter.py
│   │   │   └── rrf_fusion.py
│   │   └── use_cases/
│   │       ├── answer_query.py
│   │       ├── ingest_documents.py
│   │       └── retrieve_images.py
│   │
│   ├── infrastructure/
│   │   ├── extraction/
│   │   ├── chunking/
│   │   ├── embeddings/
│   │   ├── vector_store/
│   │   ├── bm25/
│   │   ├── checkpoint/
│   │   ├── image_store/
│   │   ├── llm/
│   │   └── logging/
│   │
│   └── interfaces/
│       └── cli/
│           └── commands.py
│
├── data/
├── images/
├── storage/
└── tests/
    ├── unit/
    └── integration/
```

This tree is representative; the exact files depend on the current checkout.

---

## 🛠️ Technology stack

| Component                | Technology                                     |
| ------------------------ | ---------------------------------------------- |
| Language                 | Python                                         |
| PDF extraction           | MinerU                                         |
| Complex table enrichment | Gemini API                                     |
| Embeddings               | Sentence Transformers                          |
| Vector store             | Qdrant local                                   |
| Lexical retrieval        | BM25 via `rank_bm25`                           |
| Retrieval fusion         | Reciprocal Rank Fusion                         |
| Local LLM                | Qwen3 8B via Ollama                            |
| Configuration            | Pydantic Settings                              |
| Interface                | Streamlit                                      |
| CLI                      | Typer                                          |
| Logging                  | Python logging with structured logging support |
| Testing                  | pytest                                         |

The embedding model and its dimension are configurable. Ensure the model configured for ingestion is the same model used for query embeddings and that the Qdrant collection dimension matches its output.

---

## 🚀 Getting started

### 1. Prerequisites

- Python version compatible with the project's dependencies
- MinerU installed and configured
- Ollama installed locally
- A Gemini API key for complex-table enrichment
- Sufficient memory and storage for the embedding model, local LLM, and document indexes

Pull the local generation model:

```bash
ollama pull qwen3:8b
```

Configure the Gemini API key using the environment variable or configuration expected by the extraction adapter. Keep API keys out of source control.

### 2. Install

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd paperlens

python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS / WSL2
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file:

```bash
# Linux / macOS / WSL2
cp .env.example .env
```

On Windows, copy `.env.example` to `.env` using File Explorer or PowerShell.

### 3. Add research papers

Place the PDFs to ingest in the configured data directory, which defaults to `./data`.

Example on Linux:

```bash
mkdir -p data
cp ~/papers/*.pdf data/
```

### 4. Ingest documents

If the CLI exposes the `ingest` command:

```bash
python main.py ingest
```

The ingestion workflow is responsible for extracting and normalizing documents, creating chunks, generating embeddings, and updating the configured indexes.

### 5. Launch the Streamlit interface

From the project root:

```bash
streamlit run streamlit_app.py
```

Enter a question about your indexed research papers. PaperLens retrieves relevant context, generates an answer using the local LLM, and displays available source references and associated figures.

### 6. Ask questions through the CLI

If the corresponding commands and options are implemented in your current CLI:

```bash
python main.py ask "What methodology does this paper use?"
```

Check the CLI help for supported image-related options:

```bash
python main.py --help
python main.py ask --help
```

---

## 🖼️ Image and table handling

### Images and figures

Extracted images are stored on disk and referenced by metadata rather than embedding binary data inside each chunk.

A representative directory structure is:

```text
images/
└── <document_id>/
    ├── <image_id>.png
    └── <image_id>.jpg
```

Image-related chunk metadata can include:

- `image_id` and `image_path`
- Caption text, when available
- Page number
- Section path
- Source document identifier and filename

Example:

```json
{
  "chunk_id": "document_0012",
  "text": "Fig. 3: Architecture of the proposed system",
  "metadata": {
    "chunk_types": ["image"],
    "section_path": ["Methodology", "Model Architecture"],
    "page_numbers": [7],
    "image_id": "image_3",
    "image_path": "images/document_id/image_3.png"
  }
}
```

### Tables

Tables retain their row-and-column representation and can be rendered as Markdown for retrieval and LLM context construction.

Example:

```markdown
| Model   | Accuracy |   F1 |
| ------- | -------: | ---: |
| Model A |    92.1% | 0.89 |
| Model B |    94.3% | 0.92 |
```

Each table is intended to remain an independent chunk, preserving the relationship between its headers, rows, and available footnotes.

---

## ⚙️ Configuration

Configuration is centralized in:

```text
app/config/settings.py
```

Values can be overridden through environment variables using the project's `PAPERLENS_` prefix.

| Variable                        | Purpose                             |
| ------------------------------- | ----------------------------------- |
| `PAPERLENS_DATA_DIR`            | Input PDF directory                 |
| `PAPERLENS_IMAGE_DIR`           | Extracted image directory           |
| `PAPERLENS_STORAGE_DIR`         | Persistent index directory          |
| `PAPERLENS_EMBEDDING_MODEL`     | Sentence-transformer model          |
| `PAPERLENS_EMBEDDING_DIMENSION` | Expected embedding dimension        |
| `PAPERLENS_LLM_MODEL`           | Ollama model name                   |
| `PAPERLENS_CHUNK_SIZE`          | Maximum text chunk size             |
| `PAPERLENS_CHUNK_OVERLAP`       | Text overlap between chunks         |
| `PAPERLENS_MIN_RELEVANCE_SCORE` | Retrieval filtering threshold       |
| `PAPERLENS_MAX_CHUNKS`          | Maximum chunks retained for context |
| `PAPERLENS_MAX_CONTEXT_TOKENS`  | Context token budget                |
| `PAPERLENS_RRF_K`               | RRF smoothing constant              |
| `PAPERLENS_TEMPERATURE`         | LLM generation temperature          |
| `PAPERLENS_NUM_CTX`             | LLM context-window setting          |
| `PAPERLENS_LOG_LEVEL`           | Logging verbosity                   |

Check `app/config/settings.py` for the actual defaults and validation rules in your checkout. Configuration examples should always match the settings implemented by the project.

Gemini-specific settings are configured separately according to the extraction adapter.

---

## 🧪 Testing

Run the automated test suite from the project root:

```bash
pytest -q
```

Run only unit tests:

```bash
pytest tests/unit -q
```

Run integration tests:

```bash
pytest tests/integration -q
```

These commands assume the corresponding test directories exist in the current checkout.

### Testing objectives

**Unit tests** can cover:

- Section hierarchy and heading detection
- Front-matter parsing
- Chunk packing, overlap, and content-type boundaries
- RRF fusion
- Retrieval filtering
- Typed identifiers and metadata conversion

**Integration tests** can cover:

- Extraction output normalization
- Image storage and path resolution
- Index persistence
- End-to-end ingestion and question answering

Run the relevant tests before treating a behavior as verified. Tests that require MinerU, Ollama, Gemini, or persisted indexes may need additional configuration or test fixtures.

---

## 🔮 Roadmap

- [ ] Persistent near-duplicate detection across ingestion runs
- [ ] Optional multimodal image embeddings with CLIP or SigLIP
- [ ] Additional document sources, including LaTeX, arXiv, HTML, and DOCX
- [ ] Parallel document ingestion
- [ ] Persistent chunk cache keyed by content hash
- [ ] Metrics export for monitoring and performance analysis
- [ ] Additional vector database adapters
- [ ] Automated table quality validation
- [ ] Extraction quality evaluation
- [ ] Semantic reranking for improved retrieval precision
- [ ] Improved source selection and citation evaluation

---

## 📜 License

MIT

---

**PaperLens — turning scientific papers into structured, traceable knowledge for retrieval-augmented generation.**
