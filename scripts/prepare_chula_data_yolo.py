import json
import zipfile
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import cv2
import pandas as pd
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "pui-nantheera/Parasitic_Egg_Detection_and_Classification_in_Microscopic_Images"
TARGET_SIZE = 800

DOWNLOADS = [
    {"file": "Chula-ParasiteEgg-11.zip", "zip_prefix": "Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/", "split": "train"},
    {"file": "Chula-ParasiteEgg-11_test.zip", "zip_prefix": "test/", "split": "val"}
]


# --- 1. MULTI-CORE LETTERBOXING ---
def process_single_image(src_path, dest_path, target_size=TARGET_SIZE):
    """Worker function: letterbox image to preserve aspect ratio."""
    try:
        img = cv2.imread(str(src_path))
        if img is not None:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            top = (max_dim - h) // 2
            bottom = max_dim - h - top
            left = (max_dim - w) // 2
            right = max_dim - w - left
            
            padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            success = cv2.imwrite(str(dest_path), resized)
            return success
    except Exception:
        pass
    return False


# --- 2. YOLO NDJSON EXPORT ---
def export_to_ndjson(df, ndjson_path, image_relative_path, split_name, dynamic_class_map):
    """Converts the dataframe into YOLO-compatible NDJSON with headers."""
    yolo_class_names = {str(k): v for k, v in dynamic_class_map.items()}
    
    header = {
        "type": "dataset",
        "task": "detect",
        "name": f"Chula_Parasite_{split_name}",
        "description": f"Unified 512x512 Letterboxed {split_name} Dataset",
        "class_names": yolo_class_names,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    
    with open(ndjson_path, 'w') as f:
        f.write(json.dumps(header) + "\n")
        
        for img_name, group in df.groupby("file_name"):
            # Clean and flatten the JSON filename
            clean_img_name = img_name.lstrip('/\\')
            flat_name = clean_img_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            final_img_name = os.path.splitext(flat_name)[0] + ".jpg"
            
            image_data = {
                "file_name": os.path.join(image_relative_path, final_img_name),
                "annotations": []
            }
            
            for _, row in group.iterrows():
                if pd.isna(row.get('category_id')):
                    continue
                
                cat_id = int(row['category_id'])
                
                w_orig = row['width']
                h_orig = row['height']
                max_dim = max(w_orig, h_orig)
                
                dx = (max_dim - w_orig) // 2
                dy = (max_dim - h_orig) // 2
                scale = TARGET_SIZE / max_dim
                
                new_xmin = (row['bbox_x'] + dx) * scale
                new_ymin = (row['bbox_y'] + dy) * scale
                new_w = row['bbox_w'] * scale
                new_h = row['bbox_h'] * scale
                
                x_center = (new_xmin + (new_w / 2)) / float(TARGET_SIZE)
                y_center = (new_ymin + (new_h / 2)) / float(TARGET_SIZE)
                w_norm = new_w / float(TARGET_SIZE)
                h_norm = new_h / float(TARGET_SIZE)
                
                image_data["annotations"].append({
                    "class_id": cat_id,
                    "bbox": [x_center, y_center, w_norm, h_norm]
                })
            
            f.write(json.dumps(image_data) + "\n")


# --- 3. MASTER PIPELINE ---
def build_dataset(target_dir):
    target_dir = Path(target_dir).resolve()
    
    dirs = {
        "train_img": target_dir / "images" / "train",
        "val_img": target_dir / "images" / "val",
        "processed": target_dir / "processed"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for entry in DOWNLOADS:
        f = entry["file"]
        split = entry["split"]
        zip_prefix = entry["zip_prefix"]
        
        final_img_dir = dirs[f"{split}_img"]
        raw_extract_dir = target_dir / f"raw_{split}"

        print(f"\nDownloading {f}...")
        zip_path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", local_dir=str(target_dir))

        print(f"Extracting raw files...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(raw_extract_dir)

        tasks = []
        for root, _, files in os.walk(raw_extract_dir):
            for file in files:
                src_path = Path(root) / file
                rel_path = src_path.relative_to(raw_extract_dir)
                clean_name = str(rel_path)[len(zip_prefix):] if str(rel_path).startswith(zip_prefix) else str(rel_path)
                
                if "irfanview" in clean_name.lower() or not clean_name:
                    continue

                if clean_name.lower().endswith('.json'):
                    shutil.copy2(src_path, target_dir / f"{split}_labels.json")
                    continue

                if clean_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # SURGICAL FIX: Strip the ghost "data/" directory if the zip injected it
                    if clean_name.startswith("data/"):
                        clean_name = clean_name[5:]
                    elif clean_name.startswith("data\\"):
                        clean_name = clean_name[5:]

                    # Replace slashes with underscores to preserve descriptive folder names
                    clean_name = clean_name.lstrip('/\\')
                    flat_name = clean_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    
                    final_name = os.path.splitext(flat_name)[0] + ".jpg"
                    dest_path = final_img_dir / final_name
                    tasks.append((src_path, dest_path))

        success_count = 0
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(process_single_image, [t[0] for t in tasks], [t[1] for t in tasks])
            for success in results:
                if success: success_count += 1

        print(f"Finished successfully writing {success_count} images.")
        print(f"Cleaning up raw files...")
        shutil.rmtree(raw_extract_dir)
        Path(zip_path).unlink()

    hf_hub_download(repo_id=REPO_ID, filename="test_labels_200.json", repo_type="dataset", local_dir=str(target_dir))
    shutil.move(str(target_dir / "test_labels_200.json"), str(target_dir / "val_labels.json"))

    print("\nGenerating YOLO NDJSON Files...")
    splits = ["train", "val"]
    
    for split in splits:
        label_file = target_dir / f"{split}_labels.json"
        if not label_file.exists():
            continue
            
        with open(label_file) as f:
            data = json.load(f)
            
        dynamic_categories = {c["id"]: c["name"] for c in data["categories"]}
        
        images_df = pd.DataFrame(data["images"])
        annotations_df = pd.DataFrame(data["annotations"])
        
        if not annotations_df.empty:
            bbox_df = pd.DataFrame(annotations_df["bbox"].tolist(), columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h"])
            annotations_df = pd.concat([annotations_df.drop(columns=["bbox"]), bbox_df], axis=1)

        df = images_df.merge(annotations_df, left_on="id", right_on="image_id", how="left")
        
        ndjson_out = dirs["processed"] / f"{split}_annotations.ndjson"
        
        export_to_ndjson(df, ndjson_out, f"images/{split}/", split, dynamic_categories)
        
        label_file.unlink()

    print("\nData processing complete.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "src", "data", "chula_yolo"))
    build_dataset(data_dir)