import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
import os

val_transform = T.Compose([
    T.Resize(550),
    T.CenterCrop(518),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(BASE_DIR, "yolov8n.pt")

yolo_model = YOLO(yolo_model_path)

def extract_embedding_with_yolo(image_bytes, model, device):

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_rgb)

    embeddings = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image_rgb[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(crop)
            tensor_crop = val_transform(pil_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor_crop)

            embeddings.append(emb)

    if len(embeddings) == 0:
        pil_image = Image.fromarray(image_rgb)
        tensor_img = val_transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            return model(tensor_img)

    embeddings = torch.cat(embeddings, dim=0)
    return torch.mean(embeddings, dim=0, keepdim=True)