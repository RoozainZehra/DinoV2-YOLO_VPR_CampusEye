from fastapi import FastAPI, HTTPException, UploadFile, File
import torch
import torch.nn.functional as F
import json
import os
from VPR.model import load_model, match_door_number_to_label
from VPR.yolo_utils import extract_embedding_with_yolo

app = FastAPI()

# ── VPR initialization ────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VPR_DIR  = os.path.join(BASE_DIR, "VPR")

print("Loading VPR model...")
model = load_model(os.path.join(VPR_DIR, "dino_vpr.pth"), device)

database_embeddings = torch.load(
    os.path.join(VPR_DIR, "embeddings.pt"), map_location=device
).to(device)

database_labels = torch.load(
    os.path.join(VPR_DIR, "labels.pt"), map_location=device
)

with open(os.path.join(VPR_DIR, "idx_to_class.json")) as f:
    idx_to_class = json.load(f)

known_locations = list(idx_to_class.values())
print("VPR system loaded successfully.")


@app.get("/")
def root():
    return {"status": "CampusEye Image Processing API is running"}


@app.post("/predict-location")
async def predict_location(file: UploadFile = File(...)):
    try:
        print("\n===== /predict-location API CALLED =====")
        contents = await file.read()
        print("Image received. Size (bytes):", len(contents))

        # ── Extract embedding + OCR (3 values now) ────────────
        print("Extracting embedding and running OCR...")
        query_emb, door_number, raw_ocr_text = extract_embedding_with_yolo(
            contents, model, device
        )
        print("Embedding shape:", query_emb.shape)
        print("Door number found:", door_number)
        print("Raw OCR text:", raw_ocr_text)

        # ── Door number match — highest priority ──────────────
        door_label, door_score = match_door_number_to_label(
            door_number, known_locations
        )

        # ── General OCR fuzzy match — medium priority ─────────
        ocr_label, ocr_score = match_door_number_to_label(
            raw_ocr_text, known_locations
        )

        # ── VPR cosine similarity — fallback ──────────────────
        print("Computing similarities...")
        similarities    = F.cosine_similarity(query_emb, database_embeddings)
        best_index      = torch.argmax(similarities).item()
        predicted_label = database_labels[best_index].item()
        vpr_location    = idx_to_class[str(predicted_label)]
        vpr_confidence  = round(similarities[best_index].item(), 4)
        print("VPR predicted location:", vpr_location)

        # ── Decision fusion ───────────────────────────────────
        if door_label and door_score >= 80:
            print(f"DOOR_OCR match: {door_label} (score {door_score})")
            print("===== PREDICTION COMPLETE =====\n")
            return {
                "location":    door_label,
                "method":      "DOOR_OCR",
                "door_number": door_number,
                "confidence":  door_score,
                "vpr_result":  vpr_location,
            }

        if ocr_label and ocr_score > 60:
            print(f"OCR_FUZZY match: {ocr_label} (score {ocr_score})")
            print("===== PREDICTION COMPLETE =====\n")
            return {
                "location":   ocr_label,
                "method":     "OCR_FUZZY",
                "confidence": ocr_score,
                "vpr_result": vpr_location,
            }

        print(f"VPR match: {vpr_location} (confidence {vpr_confidence})")
        print("===== PREDICTION COMPLETE =====\n")
        return {
            "location":   vpr_location,
            "method":     "VPR",
            "confidence": vpr_confidence,
        }

    except Exception as e:
        import traceback
        print("ERROR OCCURRED:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
