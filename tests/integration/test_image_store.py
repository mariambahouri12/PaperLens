from pathlib import Path

from app.config.settings import Settings
from app.domain.entities.image import ExtractedImage
from app.domain.value_objects.ids import DocumentId, ImageId
from app.domain.value_objects.section_path import SectionPath
from app.infrastructure.image_store.filesystem_image_store import FilesystemImageStore


def test_image_store_roundtrip(tmp_path: Path):
    settings = Settings(image_dir=tmp_path / "images", storage_dir=tmp_path / "storage")
    store = FilesystemImageStore(settings)
    img = ExtractedImage(
        image_id=ImageId("img_001"),
        document_id=DocumentId("doc_001"),
        page_number=3,
        section_path=SectionPath.of(["Results"]),
        image_path="",
    )
    store.save(img, b"\x89PNG\r\n")
    assert Path(img.image_path).exists()
    loaded = store.load(img)
    assert loaded == b"\x89PNG\r\n"
    assert store.exists("img_001")