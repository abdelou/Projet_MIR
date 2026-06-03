import torch
import numpy as np
from PIL import Image
from torchvision import models


class ViTB16Descriptor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)

        # On remplace la tête de classification par Identity.
        model.heads = torch.nn.Identity()

        self.model = model.eval().to(self.device)
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract_one(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        feat = self.model(x)
        feat = feat.cpu().numpy().astype(np.float32)[0]

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat