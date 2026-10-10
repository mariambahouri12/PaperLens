
import json
import re
import time
import urllib.request
from pathlib import Path

from datasets import load_dataset

TARGET = 30
OUTPUT_DIR = Path("qasper_test")
PDF_DIR = OUTPUT_DIR / "pdfs"
JSON_PATH = OUTPUT_DIR / "qasper_30.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)

print("Chargement de QASPER...")
ds = load_dataset("allenai/qasper", split="validation")
print(f"Documents disponibles : {len(ds)}")

selected = []
failed = []
attempts = 0
MAX_ATTEMPTS = min(len(ds), 150)

for paper in ds:
    if len(selected) >= TARGET or attempts >= MAX_ATTEMPTS:
        break

    raw_id = str(paper.get("id", "")).strip()
    arxiv_id = re.sub(r"^arXiv:\s*", "", raw_id, flags=re.IGNORECASE)

    # Accepter uniquement les identifiants arXiv classiques.
    if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv_id):
        failed.append({"id": raw_id, "reason": "ID arXiv non reconnu"})
        continue

    attempts += 1
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

    try:
        if not pdf_path.exists():
            request = urllib.request.Request(
                f"https://arxiv.org/pdf/{arxiv_id}",
                headers={"User-Agent": "PaperLensResearch/1.0"}
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()

            if not content.startswith(b"%PDF-"):
                raise ValueError("La réponse reçue n'est pas un PDF")

            pdf_path.write_bytes(content)
            time.sleep(1.5)

        qas = paper.get("qas") or []

        selected.append({
            "id": raw_id,
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "pdf_path": str(pdf_path),
            "qas": qas,
        })

        print(f"[{len(selected)}/{TARGET}] OK : {paper.get('title', arxiv_id)}")

    except Exception as exc:
        failed.append({"id": raw_id, "reason": str(exc)})
        print(f"[ÉCHEC] {arxiv_id}: {exc}")
        time.sleep(1.5)

JSON_PATH.write_text(
    json.dumps(selected, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)

(OUTPUT_DIR / "download_errors.json").write_text(
    json.dumps(failed, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n--- Résultat ---")
print(f"PDF récupérés : {len(selected)}/{TARGET}")
print(f"Questions : {sum(len(p['qas']) for p in selected)}")
print(f"Métadonnées : {JSON_PATH}")
print(f"PDF : {PDF_DIR.resolve()}")
print(f"Échecs : {len(failed)}")

if len(selected) < TARGET:
    print("Attention : moins de 30 PDF ont été récupérés.")
    print("Consulte download_errors.json pour les raisons.")
