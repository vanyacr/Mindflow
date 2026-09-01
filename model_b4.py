"""
model_b4.py — EfficientNet-B4 backbone variant with the 7-class head.

Architecture:
  - timm efficientnet_b4 (pretrained on ImageNet, feature dim = 1792)
  - Head: Linear(1792, 512) -> BN -> ReLU -> Dropout(0.4) ->
          Linear(512, 256)  -> BN -> ReLU -> Dropout(0.2) ->
          Linear(256, 7)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path

import config


class EmotionModelB4(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features   # 1792 for B4 (vs 1408 for B2)

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
        """Returns 256-dim embedding before final linear projection."""
        feat = self.backbone(x)
        for layer in list(self.head.children())[:-1]:
            feat = layer(feat)
        return feat


def load_model_b4(path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionModelB4().to(device)
    if path and Path(path).exists():
        state = torch.load(str(path), map_location=device)
        model.load_state_dict(state)
        print(f"  Loaded B4 weights from {path}")
    model.eval()
    return model, device
