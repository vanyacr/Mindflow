"""
model.py — EfficientNet-B2 backbone with a 7-class head.

Unchanged from before — no architecture changes requested for this retrain.
(B4 upgrade remains a separate optional step for later, per your roadmap.)
"""

import torch
import torch.nn as nn
import timm
import config


class EmotionModel(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features   # 1408 for B2

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    def get_embedding(self, x):
        """Returns 256-dim embedding before the final linear — useful for fusion."""
        feat = self.backbone(x)
        for layer in list(self.head.children())[:-1]:
            feat = layer(feat)
        return feat


def load_model(path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionModel().to(device)
    if path and path.exists():
        model.load_state_dict(torch.load(str(path), map_location=device))
        print(f"  Loaded weights from {path}")
    model.eval()
    return model, device
