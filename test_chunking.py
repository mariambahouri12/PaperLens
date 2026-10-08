import json
import uuid
from pathlib import Path

from app.domain.value_objects.ids import DocumentId
from app.infrastructure.chunking.config import ChunkingConfig
from app.infrastructure.chunking.hierarchical_chunker import HierarchicalChunker
from app.infrastructure.extraction.document_mapper import DocumentMapper

JSON_PATH = Path("app/infrastructure/extraction/output/test1.json")

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

document = DocumentMapper().to_document(
    data,
    DocumentId(uuid.uuid4().hex),
    Path("test1.pdf"),
)

chunker = HierarchicalChunker(ChunkingConfig(max_tokens=500, overlap_tokens=80))
chunks = chunker.chunk(document)

print(f"\n{len(chunks)} chunk(s), document_id={document.metadata.document_id}\n")

for chunk in chunks:
    m = chunk.metadata
    print(
        f"#{m.chunk_index:02d} "
        f"types={[t.value for t in m.chunk_types]} "
        f"pages={list(m.page_numbers)} "
        f"tokens={chunk.token_count}"
    )
    print(f"    section   : {m.section}")
    print(f"    subsection: {m.subsection}")

    if m.image_path:
        print(f"    image     : {m.image_path}")

    if m.table_id:
        print(f"    table_id  : {m.table_id}")

    print(f"    text      : {chunk.text[:90]!r}\n")