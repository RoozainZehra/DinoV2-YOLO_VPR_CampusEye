import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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