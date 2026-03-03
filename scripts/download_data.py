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
    {
        "file": "Chula-ParasiteEgg-11.zip",
        "extracts_as": "Chula-ParasiteEgg-11",
        "final_name": "train",
    },
    {
        "file": "Chula-ParasiteEgg-11_test.zip",
        "extracts_as": "test",
        "final_name": "test",
    },
    {
        "file": "test_labels_200.json",
        "extracts_as": None,
        "final_name": None,
    },
]


def _already_extracted(entry):
    """Check if data exists under either the original or final name."""
    for name in (entry["final_name"], entry["extracts_as"]):
        if name and (DATA_DIR / name).exists():
            return True
    return False


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for entry in DOWNLOADS:
        f = entry["file"]
        dest = DATA_DIR / f

        if _already_extracted(entry):
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
        if entry["extracts_as"] is None:
            continue

        zip_path = DATA_DIR / f

        if _already_extracted(entry):
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


def reorganize():
    # Flatten Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/ → train/
    old = DATA_DIR / "Chula-ParasiteEgg-11" / "Chula-ParasiteEgg-11"
    new = DATA_DIR / "train"
    if old.exists() and not new.exists():
        old.rename(new)
        (DATA_DIR / "Chula-ParasiteEgg-11").rmdir()
        print("Renamed Chula-ParasiteEgg-11/Chula-ParasiteEgg-11 → train")


def cleanup():
    junk = [
        DATA_DIR / ".cache",
        DATA_DIR / "test" / "data" / "irfanview",
    ]
    for path in junk:
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {path.relative_to(DATA_DIR)}")

    for entry in DOWNLOADS:
        zip_path = DATA_DIR / entry["file"]
        if zip_path.suffix == ".zip" and zip_path.exists():
            zip_path.unlink()
            print(f"Removed {zip_path.name}")


def verify():
    counts = {}
    for entry in DOWNLOADS:
        final = entry["final_name"]
        if final is None:
            continue
        extract_dir = DATA_DIR / final
        images = list(extract_dir.rglob("*.jpg"))
        counts[final] = len(images)
        print(f"  {final}: {len(images)} images")
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
    reorganize()
    cleanup()
    print()
    print("Verification:")
    counts = verify()
    write_metadata(counts)
    print("\nDone.")


if __name__ == "__main__":
    main()
