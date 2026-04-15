import os
import yaml
import json
import torch
from ultralytics import YOLO
from email_complete import send_training_notification
import glob
import time


""" DO NOT CHANGE: SET BY DATA PREPROCESSING CHOICE """
""" ----------------------------------------------- """
IMGSZ = 800
""" ----------------------------------------------- """


def unpack_ndjson_and_build_yaml(data_dir, yaml_path):
    """
    read master NDJSON files and  unpack them into yolo text format.
    skip the unpacking process if the labels already exist.
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
    n_epochs = 25
    batch_size = 16
    n_workers = 4
    run_name = 'chula_training'
    send_email = True
    terminate_pod = True

    # paths for data, yaml, weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "src", "data", "chula_yolo"))
    runs_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments", "yolo_training"))
    yaml_path = os.path.abspath(os.path.join(script_dir, "..", "src", "data", "chula_yolo", "dataset_config.yaml"))


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
        if os.path.exists(os.path.abspath(os.path.join(script_dir, "yolo26l.pt"))):
            model = YOLO(os.path.abspath(os.path.join(script_dir, "yolo26l.pt")))
        else:
            model = YOLO("yolo26l.pt")

    # capture the time first epoch starts (ignoring data loading)
    epoch_start_time = [0]
    def capture_start_time(trainer):
        epoch_start_time[0] = time.time()
    model.add_callback("on_train_start", capture_start_time)


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
        cache=False,
        mosaic=0,              # ALWAYS KEEP AT 0
        mixup=0,               # ALWAYS KEEP AT 0
        fliplr=0.5,                 
        flipud=0.5,                 
        scale=0.7,           
        cos_lr=True,         
        patience=20          
    )


    ## Send email notifying completion of training, best fitness, avg epoch time
    if send_email:
        # compute time metrics
        total_train_time = time.time() - epoch_start_time[0]
        avg_epoch_seconds = total_train_time / n_epochs
        avg_mins = int(avg_epoch_seconds // 60)

        best_fitness = model.trainer.best_fitness

        message = (
            f"Training complete.\n"
            f"Best Fitness: {best_fitness:.4f}\n"
            f"Average Epoch Time: {avg_mins}m"
        )

        try:
            send_training_notification("Training complete", message)
            print("Notification email sent successfully.")
        except Exception as e:
            print(f"Warning: Could not send email notification. {e}")
        

    ## Terminate runpod upon completion of training
    if terminate_pod:
        import runpod
        pod_id = os.environ.get("RUNPOD_POD_ID")
        key_path = "/workspace/.runpod_key"
    
        if pod_id and os.path.exists(key_path):
            with open(key_path, 'r') as f:
                runpod.api_key = f.read().strip()
            print(f"Terminating pod: {pod_id}")
            runpod.terminate_pod(pod_id)
        else:
            print("Could not auto-terminate: missing RUNPOD_POD_ID or API key file.")
            if send_email:
                send_training_notification("Could not terminate pod", "Could not auto-terminate: missing RUNPOD_POD_ID or API key file.")


if __name__ == "__main__":
    main()