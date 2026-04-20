import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from rapidfuzz import process, fuzz

class DINOv2_VPR(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False
        )
        self.backbone.reset_classifier(0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embedding_head = nn.Linear(384, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def load_model(weight_path, device):
    model = DINOv2_VPR().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


# ── Door number extraction ────────────────────────────────────
DOOR_NUMBER_PATTERNS = [
    r'\b([A-Z]{1,3}[-–]?\d{1,4}[A-Z]?)\b',
    r'\b([A-Z]{1,3}\d{1,2}[-–]\d{1,4})\b',
    r'\b(?:Room|Lab|Hall|Block)\s*(\d{1,4}[A-Z]?)\b',
    r'\b(\d{1,4}[A-Z]?)\b',
]

def extract_door_number(raw_text: str):
    if not raw_text:
        return None
    text = re.sub(r'\s+', ' ', raw_text).upper().strip()
    for pattern in DOOR_NUMBER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return re.sub(r'[-–\s]', '', matches[0]).upper()
    return None


def match_door_number_to_label(door_number: str, known_locations: list):
    if not door_number:
        return None, 0
    normalised_labels = {
        re.sub(r'[-–\s]', '', lbl).upper(): lbl
        for lbl in known_locations
    }
    result = process.extractOne(
        door_number,
        list(normalised_labels.keys()),
        scorer=fuzz.ratio
    )
    if result is None:
        return None, 0
    match_norm, score, _ = result
    return normalised_labels[match_norm], score
