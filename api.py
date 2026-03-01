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
VPR_DIR = os.path.join(BASE_DIR, "VPR")

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


# check server is up
@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "CampusEye Image Processing API is running"}


# VPR API call
@app.post("/predict-location")
async def predict_location(file: UploadFile = File(...)):
    try:
        print("\n===== /predict-location API CALLED =====")

        contents = await file.read()
        print("Image received. Size (bytes):", len(contents))

        # Extract embedding using YOLO + DINOv2
        print("Extracting embedding...")
        query_emb = extract_embedding_with_yolo(contents, model, device)
        print("Embedding shape:", query_emb.shape)

        # Cosine similarity
        print("Computing cosine similarity...")
        similarities = F.cosine_similarity(query_emb, database_embeddings)
        print("Similarity scores:", similarities)

        best_index = torch.argmax(similarities).item()
        print("Best match index:", best_index)

        predicted_label = database_labels[best_index].item()
        print("Predicted label index:", predicted_label)

        predicted_location = idx_to_class[str(predicted_label)]
        print("Predicted location:", predicted_location)

        print("===== PREDICTION COMPLETE =====\n")

        return {"location": predicted_location}

    except Exception as e:
        import traceback
        print("ERROR OCCURRED:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))