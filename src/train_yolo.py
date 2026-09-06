import h5py
import numpy as np
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO

def extract_and_format_dataset(hdf5_path, output_dir):
    """
    Extrai as sequências do HDF5 do dvs-benchmark-2021 e converte 
    para a estrutura padrão de dataset exigida pelo YOLOv8.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images/train"
    labels_dir = output_dir / "labels/train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo arquivo de estudo: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        # Extração de frames da câmera Photron
        photron_data = f['photron']
        frames = photron_data['frames'][:]
        bboxes = photron_data['bounding_boxes'][:]  # Format: [x, y, w, h]

        for idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
            img_name = f"frame_{idx:05d}.png"
            img_path = images_dir / img_name
            cv2.imwrite(str(img_path), frame)

            # Normalização de Bounding Box para formato YOLO (cx, cy, w, h relativos [0, 1])
            h_img, w_img = frame.shape[:2]
            x, y, w, h = bbox
            if w > 0 and h > 0:
                cx = (x + w / 2.0) / w_img
                cy = (y + h / 2.0) / h_img
                nw = w / w_img
                nh = h / h_img

                label_path = labels_dir / f"frame_{idx:05d}.txt"
                with open(label_path, "w") as lf:
                    # Classe 0: projectile
                    lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Criar arquivo data.yaml do YOLO
    yaml_data = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/train',  # PoC utilizando a sequência do benchmark para validação
        'names': {0: 'projectile'}
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as yf:
        yaml.dump(yaml_data, yf, default_flow_style=False)

    return yaml_path

def train_yolo_model(data_yaml_path, epochs=15, img_size=640):
    """
    Treina a arquitetura YOLOv8n nos frames de balística.
    """
    model = YOLO("yolov8n.pt")  # Carrega o modelo pré-treinado base
    print("Iniciando treinamento do modelo YOLOv8n...")
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        name="yolo_ballistic_poc",
        project="models",
        exist_ok=True
    )
    return results

if __name__ == "__main__":
    HDF5_FILE = "../ballistic_experiments.hdf5"
    DATASET_DIR = "data/yolo_dataset"

    yaml_file = extract_and_format_dataset(HDF5_FILE, DATASET_DIR)
    train_yolo_model(yaml_file, epochs=15)
