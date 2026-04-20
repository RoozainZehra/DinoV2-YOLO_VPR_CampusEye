import re
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
import easyocr
import os

from VPR.model import extract_door_number

val_transform = T.Compose([
    T.Resize(550),
    T.CenterCrop(518),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
yolo_model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def preprocess_for_ocr(image_bgr):
    variants = []
    gray     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)
    variants.append(enhanced)

    _, thresh_otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(thresh_otsu)
    variants.append(cv2.bitwise_not(thresh_otsu))

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    variants.append(adaptive)

    h, w = gray.shape
    if max(h, w) < 200:
        scale    = 200 / max(h, w)
        upscaled = cv2.resize(
            enhanced, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
        variants.append(upscaled)

    return variants


def ocr_crop(crop_bgr):
    all_texts = []
    for variant in preprocess_for_ocr(crop_bgr):
        results = ocr_reader.readtext(variant, detail=1)
        for (_, text, conf) in results:
            if conf > 0.2:
                all_texts.append(text)
    combined = " ".join(all_texts)
    return extract_door_number(combined), combined


def extract_embedding_with_yolo(image_bytes, model, device):
    np_arr    = np.frombuffer(image_bytes, np.uint8)
    image     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_rgb)

    embeddings    = []
    all_door_nums = []
    all_raw_texts = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop_rgb = image_rgb[y1:y2, x1:x2]
            crop_bgr = image[y1:y2, x1:x2]

            if crop_rgb.size == 0:
                continue

            door_num, raw_text = ocr_crop(crop_bgr)
            if door_num:
                all_door_nums.append(door_num)
            all_raw_texts.append(raw_text)

            pil_crop    = Image.fromarray(crop_rgb)
            tensor_crop = val_transform(pil_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor_crop)
            embeddings.append(emb)

    # Fallback: full image if YOLO found nothing
    if not embeddings:
        door_num, raw_text = ocr_crop(image)
        if door_num:
            all_door_nums.append(door_num)
        all_raw_texts.append(raw_text)

        pil_image  = Image.fromarray(image_rgb)
        tensor_img = val_transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor_img)
        embeddings.append(emb)

    embeddings      = torch.cat(embeddings, dim=0)
    final_embedding = torch.mean(embeddings, dim=0, keepdim=True)
    best_door_num   = all_door_nums[0] if all_door_nums else None
    combined_text   = " ".join(all_raw_texts)

    return final_embedding, best_door_num, combined_text
