import json
import uuid
from dataclasses import asdict
from pathlib import Path

from app.domain.value_objects.ids import DocumentId
from app.infrastructure.chunking.config import ChunkingConfig
from app.infrastructure.chunking.hierarchical_chunker import HierarchicalChunker
from app.infrastructure.extraction.document_mapper import DocumentMapper

OUTPUT_DIR = Path("app/infrastructure/extraction/output")
JSON_PATH = OUTPUT_DIR / "test1.json"
CHUNKS_PATH = OUTPUT_DIR / "test1_chunks.json"

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

document = DocumentMapper().to_document(
    data,
    DocumentId(uuid.uuid4().hex),
    Path("test1.pdf"),
)

chunker = HierarchicalChunker(ChunkingConfig(max_tokens=500, overlap_tokens=80))
chunks = chunker.chunk(document)


def serialize(chunk) -> dict:
    m = chunk.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "metadata": {
            "document_id": m.document_id,
            "filename": m.filename,
            "chunk_index": m.chunk_index,
            "types": [t.value for t in m.chunk_types],
            "section": m.section,
            "subsection": m.subsection,
            "section_path": m.section_path.as_list(),
            "pages": list(m.page_numbers),
            "image_id": m.image_id,
            "image_path": m.image_path,
            "table_id": m.table_id,
        },
    }


CHUNKS_PATH.write_text(
    json.dumps([serialize(c) for c in chunks], ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"{len(chunks)} chunk(s) written to {CHUNKS_PATH}\n")

for chunk in chunks:
    m = chunk.metadata
    print(
        f"#{m.chunk_index:02d} types={[t.value for t in m.chunk_types]} "
        f"pages={list(m.page_numbers)} tokens={chunk.token_count} "
        f"| {m.section} | {m.subsection}"
    )