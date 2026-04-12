import os
import yaml
import json
import torch
from ultralytics import YOLO
from email_complete import send_training_notification
import glob


""" DO NOT CHANGE: SET BY DATA PREPROCESSING CHOICE """
IMGSZ = 800
""" ----------------------------------------------- """


def unpack_ndjson_and_build_yaml(data_dir, yaml_path):
    """
    reads master NDJSON files and  unpack them into yolo text format.
    will skip the unpacking process if the labels already exist.
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


def create_email_callback(send_email=False):
    # generate callback function for end of yolo training

    def callback(trainer):
        # empty callback function
        if not send_email:
            return

        # full callback function
        best_fitness = trainer.best_fitness 
        save_dir = trainer.save_dir
        message = f"Training complete.\nBest Fitness: {best_fitness:.4f}\nWeights saved to: {save_dir}/weights"
        
        try:
            send_training_notification("Training complete", message)
        except Exception as e:
            print(f"Warning: Could not send email notification. {e}")
            
    # return the callback function
    return callback


def get_last_best_weights(runs_dir):
    """
    to save the hassle of having to hard-code the best weights of the previous run,
    this function searches through the master directory and pulls the last-saved best weights.
    """

    if not os.path.exists(runs_dir):
        return None
        
    # search for best.pt files inside any subfolder in the runs directory
    search_pattern = os.path.join(runs_dir, "*", "weights", "best.pt")
    all_weights = glob.glob(search_pattern)
    
    if not all_weights:
        return None
        
    # return the file with the most recent modification timestamp
    latest_weights = max(all_weights, key=os.path.getmtime)
    return latest_weights


def main():
    # parameters
    n_epochs = 3
    batch_size = 32
    n_workers = 16
    run_name = 'chula_training'
    send_email = False

    # paths for data, yaml, weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "src", "data", "chula_yolo"))
    runs_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments", "yolo_runs"))
    yaml_path = os.path.abspath(os.path.join(script_dir, "..", "configs", "dataset_config.yaml"))


    # check for gpu
    if torch.cuda.is_available():
        target_device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device {gpu_name}.")
        # Turn on the Tensor Cores
        torch.set_float32_matmul_precision('high')
    else:
        target_device = 'cpu'
        print("No GPU found. Falling back to CPU training (extremely slow).")


    print("Preparing dataset...")
    unpack_ndjson_and_build_yaml(data_dir, yaml_path)


    print("Initializing model...")
    # check through previous weights and initialize model with last-saved best weights
    best_weights = get_last_best_weights(runs_dir)
    if best_weights is not None:
        print(f"Using last-saved best weights.")
        model = YOLO(best_weights)
    else:
        print("No weights found. Loading base weights.")
        model = YOLO("yolo26l.pt")


    # email upon completing training if desired
    model.add_callback("on_train_end", create_email_callback(send_email=send_email))


    print("Training...")
    results = model.train(
        data=yaml_path,
        epochs=n_epochs,
        imgsz=IMGSZ,
        batch=batch_size,            
        device=target_device,
        workers=n_workers,           
        project=runs_dir, 
        name=run_name,
        cache=True,
        mosaic=1.0,          
        mixup=0.1,           
        scale=0.7,           
        cos_lr=True,         
        patience=20          
    )

if __name__ == "__main__":
    main()