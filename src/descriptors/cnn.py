import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms


class ResNet50Descriptor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        # on enlève la couche fully-connected finale
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)

        self.transform = weights.transforms()

    @torch.no_grad()
    def extract_one(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        feat = self.model(x)
        feat = feat.flatten(1)

        feat = feat.cpu().numpy().astype(np.float32)[0]

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat