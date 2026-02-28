from fastapi import FastAPI, HTTPException, UploadFile, File


#VPR dependencies
import torch
import torch.nn.functional as F
import json
import os

from VPR.model import load_model
from VPR.yolo_utils import extract_embedding_with_yolo

app = FastAPI()


#VPR INITIALIZATION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VPR_DIR = os.path.join(BASE_DIR, "vpr")

print("Loading VPR model...")

model = load_model(os.path.join(VPR_DIR, "dino_vpr.pth"), device)

database_embeddings = torch.load(
    os.path.join(VPR_DIR, "embeddings.pt"),
    map_location=device
).to(device)

database_labels = torch.load(
    os.path.join(VPR_DIR, "labels.pt"),
    map_location=device
)

with open(os.path.join(VPR_DIR, "idx_to_class.json")) as f:
    idx_to_class = json.load(f)

print("VPR system loaded successfully.")


# VPR API call
@app.post("/predict-location")
async def predict_location(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Extract embedding using YOLO + DINOv2
        query_emb = extract_embedding_with_yolo(contents, model, device)

        # Cosine similarity
        similarities = F.cosine_similarity(query_emb, database_embeddings)

        best_index = torch.argmax(similarities).item()

        predicted_label = database_labels[best_index].item()

        predicted_location = idx_to_class[str(predicted_label)]
        
        return {"location": predicted_location}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))