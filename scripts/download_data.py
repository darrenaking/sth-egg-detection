"""
Download and extract Chula-ParasiteEgg-11 dataset from HuggingFace.
Usage: python scripts/download_data.py
"""

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "pui-nantheera/Parasitic_Egg_Detection_and_Classification_in_Microscopic_Images"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DOWNLOADS = [
    {"file": "Chula-ParasiteEgg-11.zip", "extract_to": "Chula-ParasiteEgg-11"},
    {"file": "Chula-ParasiteEgg-11_test.zip", "extract_to": "test"},
    {"file": "test_labels_200.json", "extract_to": None},
]


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for entry in DOWNLOADS:
        f = entry["file"]
        dest = DATA_DIR / f
        extracted = entry["extract_to"]

        if extracted and (DATA_DIR / extracted).exists():
            print(f"Already extracted, skipping download: {f}")
            continue
        if dest.exists():
            print(f"Already downloaded, skipping: {f}")
            continue

        print(f"Downloading {f}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
    print("Downloads complete.")


def extract():
    for entry in DOWNLOADS:
        f = entry["file"]
        extracted = entry["extract_to"]
        if extracted is None:
            continue

        zip_path = DATA_DIR / f
        extract_dir = DATA_DIR / extracted

        if extract_dir.exists():
            print(f"Already extracted, skipping: {f}")
            continue
        if not zip_path.exists():
            print(f"Zip not found, skipping: {f}")
            continue

        print(f"Extracting {f}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)

        zip_path.unlink()
        print(f"Deleted {f}")

    print("Extraction complete.")


def cleanup():
    junk = [
        DATA_DIR / ".cache",
        DATA_DIR / "test" / "data" / "irfanview",
    ]
    for path in junk:
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {path.relative_to(DATA_DIR)}")


def verify():
    counts = {}
    for entry in DOWNLOADS:
        extracted = entry["extract_to"]
        if extracted is None:
            continue
        extract_dir = DATA_DIR / extracted
        images = list(extract_dir.rglob("*.jpg"))
        counts[extracted] = len(images)
        print(f"  {extracted}: {len(images)} images")
    return counts


def write_metadata(counts):
    meta = {
        "repo_id": REPO_ID,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "image_counts": counts,
    }
    meta_path = DATA_DIR / "download_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path.name}")


def main():
    print(f"Data directory: {DATA_DIR}\n")
    download()
    print()
    extract()
    cleanup()
    print()
    print("Verification:")
    counts = verify()
    write_metadata(counts)
    print("\nDone.")


if __name__ == "__main__":
    main()
