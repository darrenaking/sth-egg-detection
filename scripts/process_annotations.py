"""
Process COCO-format annotations into flat CSVs for analysis.
Usage: python scripts/process_annotations.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = {
    "train": {
        "labels": RAW_DIR / "train" / "labels.json",
        "image_dir": "data/raw/train/data",
    },
    "test": {
        "labels": RAW_DIR / "test_labels_200.json",
        "image_dir": "data/raw/test/data",
    },
}

BBOX_COLUMNS = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]

NULLABLE_INT_COLS = {
    "annotation_id": "Int64",
    "category_id": "Int64",
}

FLOAT_COLS = {
    "bbox_x": "float64",
    "bbox_y": "float64",
    "bbox_w": "float64",
    "bbox_h": "float64",
    "area": "float64",
}


def load_coco(path):
    with open(path) as f:
        data = json.load(f)
    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = pd.DataFrame(data["images"])
    annotations = pd.DataFrame(data["annotations"])
    return images, annotations, categories


def build_dataframe(images, annotations, categories, image_dir):
    # Unpack bbox list into columns
    if not annotations.empty:
        bbox_df = pd.DataFrame(
            annotations["bbox"].tolist(), columns=BBOX_COLUMNS
        )
        annotations = pd.concat(
            [annotations.drop(columns=["bbox"]), bbox_df], axis=1
        )
        annotations["category_name"] = annotations["category_id"].map(categories)
        annotations = annotations.rename(columns={"id": "annotation_id"})

    # Left join to keep images with no annotations
    df = images.merge(
        annotations,
        left_on="id",
        right_on="image_id",
        how="left",
    )

    # Drop redundant/empty columns
    df = df.drop(columns=["image_id", "coco_url", "license"], errors="ignore")
    df = df.rename(columns={"id": "image_id"})

    # Enforce dtypes to prevent NaN coercion of int columns
    for col, dtype in {**NULLABLE_INT_COLS, **FLOAT_COLS}.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # Add relative file path from project root
    df["file_path"] = image_dir + "/" + df["file_name"]

    # Deterministic output
    df = df.sort_values(["image_id", "annotation_id"]).reset_index(drop=True)

    return df


def check_filename_classes(df):
    """Training only: verify annotation labels match the class encoded in filenames."""
    df["filename_class"] = df["file_name"].str.rsplit("_", n=1).str[0]
    annotated = df.dropna(subset=["category_name"])
    mismatches = annotated[annotated["filename_class"] != annotated["category_name"]]
    if len(mismatches) > 0:
        print(f"  WARNING: {len(mismatches)} filename/annotation mismatches!")
        print(mismatches[["file_name", "filename_class", "category_name"]].head(10))
    else:
        print("  Filename/annotation class check passed.")
    return df


def print_summary(df, split_name):
    n_images = df["image_id"].nunique()
    n_annotations = int(df["annotation_id"].notna().sum())
    n_unannotated = int(df["category_name"].isna().sum())

    print(f"\n  {split_name}: {n_images} images, {n_annotations} annotations")

    resolutions = df.groupby("image_id")[["width", "height"]].first()
    unique_res = resolutions.drop_duplicates()
    print(f"  {len(unique_res)} unique resolutions")
    for _, row in unique_res.iterrows():
        count = ((resolutions["width"] == row["width"]) & (resolutions["height"] == row["height"])).sum()
        print(f"    {int(row['width'])}x{int(row['height'])}: {count} images")

    if n_unannotated > 0:
        print(f"  {n_unannotated} images with no annotations")

    print("  Per-category counts:")
    counts = df["category_name"].value_counts().sort_index()
    for name, count in counts.items():
        print(f"    {name}: {count}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, config in SPLITS.items():
        print(f"Processing {split_name}...")
        images, annotations, categories = load_coco(config["labels"])
        df = build_dataframe(images, annotations, categories, config["image_dir"])

        if split_name == "train":
            df = check_filename_classes(df)

        print_summary(df, split_name)

        out_path = PROCESSED_DIR / f"{split_name}_annotations.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path.relative_to(PROJECT_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
