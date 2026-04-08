import os
import yaml
import json
import torch
from ultralytics import YOLO
from email_complete import send_training_notification

def unpack_ndjson_and_build_yaml(data_dir, yaml_path):
    """
    Reads the master NDJSONs and safely unpacks them into YOLO text format.
    Will skip the unpacking process if the labels already exist.
    """
    train_ndjson = os.path.join(data_dir, "processed", "train_annotations.ndjson")
    val_ndjson = os.path.join(data_dir, "processed", "val_annotations.ndjson")
    
    # 1. Read Header to get the dynamic dictionary
    with open(train_ndjson, 'r') as f:
        header = json.loads(f.readline())
        
    yolo_class_names = {int(k): v for k, v in header.get("class_names", {}).items()}
    
    # 2. Unpack NDJSON into YOLO .txt files (ONLY if needed)
    for split, ndjson_path in [("train", train_ndjson), ("val", val_ndjson)]:
        labels_dir = os.path.join(data_dir, "labels", split)
        
        # Check if the folder exists and already has files in it
        if os.path.exists(labels_dir) and len(os.listdir(labels_dir)) > 0:
            continue
            
        os.makedirs(labels_dir, exist_ok=True)
        
        if not os.path.exists(ndjson_path):
            continue
            
        with open(ndjson_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines[1:]: # Skip the header line
            data = json.loads(line)
            
            # Safely extract the base filename
            base_name = os.path.splitext(os.path.basename(data["file_name"]))[0]
            txt_path = os.path.join(labels_dir, base_name + ".txt")
            
            # Write the YOLO formatted line
            with open(txt_path, 'w') as txt_f:
                for ann in data.get("annotations", []):
                    bbox = ann["bbox"]
                    txt_f.write(f"{ann['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    # 3. Build Standard YOLO YAML pointing to the actual Image Directories
    config = {
        'path': data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': yolo_class_names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    return yaml_path

def main():
    # --- DYNAMIC PATHING ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "chula_yolo"))
    runs_dir = os.path.abspath(os.path.join(script_dir, "..", "runs", "parasite_detection"))
    yaml_path = os.path.join(data_dir, "dataset_config.yaml")

    # --- HARDWARE CHECK ---
    if torch.cuda.is_available():
        target_device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Check: SUCCESS. Found GPU -> {gpu_name}")
        # Turn on the Tensor Cores
        torch.set_float32_matmul_precision('high')
    else:
        target_device = 'cpu'
        print("Hardware Check: WARNING. No GPU found. Falling back to CPU training (This will be extremely slow).")

    print("1. Preparing YOLO Dataset...")
    unpack_ndjson_and_build_yaml(data_dir, yaml_path)

    print("2. Initializing YOLO26 Model...")
    model = YOLO("yolo26s.pt")

    # --- CUSTOM EMAIL CALLBACK ---
    def on_train_end(trainer):
        best_fitness = trainer.best_fitness 
        save_dir = trainer.save_dir
        message = f"Training complete!\nBest Fitness: {best_fitness:.4f}\nWeights saved to: {save_dir}/weights/best.pt"
        try:
            send_training_notification("YOLO26 Training Complete", message)
        except Exception as e:
            print(f"Warning: Could not send email notification. {e}")

    model.add_callback("on_train_end", on_train_end)

    print("3. Training...")
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=800,
        batch=32,            
        device=target_device,
        workers=16,           
        cache=False,         
        project=runs_dir, 
        name='chula_run_2_nocache',
        mosaic=1.0,          
        mixup=0.1,           
        scale=0.7,           
        cos_lr=True,         
        patience=20          
    )

if __name__ == "__main__":
    main()