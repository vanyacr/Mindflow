"""
model_temporal.py — Track B temporal model.

Architecture: frozen Track A EfficientNet-B2 (already validated at 74.6%
static accuracy) used as a per-frame feature extractor, feeding a BiGRU
that models motion/expression change across the clip.

Why frozen backbone instead of training a temporal CNN from scratch:
  - Reuses a checkpoint already proven to work
  - Far fewer trainable params -> much less prone to overfitting on
    DFEW's smaller classes (disgust: 116 train clips) than a from-scratch
    spatio-temporal CNN would be
  - This is the direct architectural fix for the overfitting that paused
    Track B previously — fewer parameters chasing the same small data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_temporal as cfg
from model import EmotionModel, load_model


class TemporalEmotionModel(nn.Module):
    def __init__(self, static_ckpt_path=None, num_classes: int = cfg.NUM_CLASSES,
                 hidden_size: int = cfg.GRU_HIDDEN, num_layers: int = cfg.GRU_LAYERS,
                 dropout: float = cfg.HEAD_DROPOUT):
        super().__init__()

        static_ckpt_path = static_ckpt_path or cfg.STATIC_CKPT

        # ── frozen per-frame feature extractor (Track A backbone) ──
        # By default fully frozen. unfreeze_last_block() can optionally
        # unfreeze just the backbone's final block for limited domain
        # adaptation — see train_temporal.py's --unfreeze_backbone flag.
        self.static_model, _ = load_model(static_ckpt_path, device=torch.device("cpu"))
        for p in self.static_model.parameters():
            p.requires_grad = False
        self.static_model.eval()

        embed_dim = 256   # EmotionModel.get_embedding() output size

        # ── BiGRU over the sequence of frame embeddings ──
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else 0.2,
        )

        gru_out_dim = hidden_size * 2   # bidirectional

        # ── attention pooling over timesteps ──
        # Diagnostic finding: mean-pooling dilutes the neutral class (flat,
        # low-signal clips get averaged together with incidental per-frame
        # noise) while the static per-frame baseline stayed robust to that
        # same noise. Attention pooling lets the model learn which frames
        # in a clip actually carry emotion signal instead of weighting
        # all 16 equally.
        self.attn = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # ── learnable residual scaling for temporal delta ──
        # Anchors predictions to the proven static frame-averaged baseline (which
        # already gets 60.8% neutral and 54.5% sad), while allowing the BiGRU to
        # learn dynamic expression offsets (e.g., surprise +14.1%, happy +7.1%).
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def unfreeze_last_block(self):
        """
        Bounded domain-adaptation experiment: unfreeze only the final
        block of the EfficientNet-B2 backbone (timm's efficientnet_b2
        exposes blocks as backbone.blocks, an nn.Sequential of stages).
        Everything else stays frozen. Called explicitly, opt-in only —
        default behaviour is fully frozen.
        """
        last_block = self.static_model.backbone.blocks[-1]
        for p in last_block.parameters():
            p.requires_grad = True
        n_unfrozen = sum(p.numel() for p in last_block.parameters())
        print(f"  Unfroze backbone's last block: {n_unfrozen:,} params now trainable")

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) -> (B, T, embed_dim)
        Uses no_grad only if the backbone is fully frozen — if the last
        block was unfrozen via unfreeze_last_block(), gradients need to
        flow through for those params to actually train.
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        backbone_trainable = any(p.requires_grad for p in self.static_model.parameters())
        if backbone_trainable:
            emb = self.static_model.get_embedding(x)
        else:
            with torch.no_grad():
                emb = self.static_model.get_embedding(x)
        return emb.view(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embeddings(x)          # (B, T, 256)

        # ── (1) static per-frame classification projection ──
        # Linear(256, 7) from the proven Track A static head
        static_logits = self.static_model.head[-1](emb)   # (B, T, 7)
        static_avg = static_logits.mean(dim=1)             # (B, 7)

        # ── (2) BiGRU over sequence of embeddings ──
        gru_out, _ = self.gru(emb)                 # (B, T, hidden*2)

        # ── (3) attention pooling over timesteps ──
        attn_scores = self.attn(gru_out)            # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, T, 1)
        pooled = (gru_out * attn_weights).sum(dim=1)        # (B, hidden*2)

        # ── (4) temporal delta head ──
        temp_delta = self.head(pooled)                      # (B, 7)

        # ── (5) residual static + temporal fusion ──
        return static_avg + self.residual_scale * temp_delta

    def train(self, mode: bool = True):
        """
        Override so a FULLY frozen backbone always stays in eval mode
        (stable BatchNorm stats), even when model.train() is called on
        the whole module. If unfreeze_last_block() was called, the
        backbone is allowed into train mode too, so BatchNorm/dropout
        in that last block actually update during training.
        """
        super().train(mode)
        backbone_trainable = any(p.requires_grad for p in self.static_model.parameters())
        if not backbone_trainable:
            self.static_model.eval()
        else:
            self.static_model.train(mode)
        return self


class TemporalTransformerEmotionModel(nn.Module):
    """
    Spatial-Temporal Transformer Model for Video FER (Former-DFER inspired).

    1. Extracts full 1408-dimensional spatial feature maps directly from the EfficientNet-B2 backbone.
    2. Projects 1408 -> d_model (512) with LayerNorm and Dropout.
    3. Adds learnable temporal positional embeddings across the 16 timesteps.
    4. Applies a 2-layer Multi-Head Self-Attention Transformer Encoder (8 attention heads, d_ff=1024, GELU).
    5. Applies learned multi-head temporal attention pooling.
    6. Combines with static frame-averaged prediction via learnable residual scaling.
    """
    def __init__(self, static_ckpt_path=None, num_classes: int = cfg.NUM_CLASSES,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 2,
                 dim_feedforward: int = 1024, dropout: float = 0.2,
                 seq_len: int = cfg.SEQ_LEN):
        super().__init__()
        static_ckpt_path = static_ckpt_path or cfg.STATIC_CKPT
        self.static_model, _ = load_model(static_ckpt_path, device=torch.device("cpu"))
        for p in self.static_model.parameters():
            p.requires_grad = False
        self.static_model.eval()

        backbone_dim = self.static_model.backbone.num_features  # 1408 for B2

        # ── Feature Projection ──
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # ── Learnable Positional Encoding ──
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Learned Temporal Attention Pooling ──
        self.attn = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # ── Temporal Classification Head ──
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # ── Learnable Residual Scaling ──
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> (B, T, 1408)"""
        B, T, C, H, W = x.shape
        flat = x.view(B * T, C, H, W)
        with torch.no_grad():
            feats = self.static_model.backbone(flat)  # (B*T, 1408)
        return feats.view(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # (1) Static per-frame prediction anchor
        flat = x.view(B * T, C, H, W)
        with torch.no_grad():
            static_logits = self.static_model(flat)      # (B*T, 7)
            static_probs = F.softmax(static_logits, dim=1).view(B, T, -1)
            static_avg_probs = static_probs.mean(dim=1)  # (B, 7)
            static_anchor = torch.log(static_avg_probs + 1e-7)

        # (2) Extract raw 1408-dim backbone features
        feats = self.extract_features(x)                # (B, T, 1408)

        # (3) Project and add positional embeddings
        h = self.proj(feats) + self.pos_embed           # (B, T, 512)

        # (4) Transformer multi-head temporal self-attention
        tokens = self.transformer(h)                    # (B, T, 512)

        # (5) Attention pooling over time
        attn_weights = torch.softmax(self.attn(tokens), dim=1)  # (B, T, 1)
        pooled = (tokens * attn_weights).sum(dim=1)             # (B, 512)

        # (6) Temporal delta
        delta = self.head(pooled)                               # (B, 7)

        return static_anchor + self.residual_scale * delta

    def train(self, mode: bool = True):
        super().train(mode)
        self.static_model.eval()  # backbone always stays in eval mode
        return self


def load_temporal_model(ckpt_path=None, device=None, is_transformer: bool = False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_transformer:
        model = TemporalTransformerEmotionModel().to(device)
        if ckpt_path and Path(ckpt_path).exists():
            state = torch.load(str(ckpt_path), map_location=device)
            model.proj.load_state_dict(state["proj"])
            model.pos_embed.data.copy_(state["pos_embed"])
            model.transformer.load_state_dict(state["transformer"])
            model.attn.load_state_dict(state["attn"])
            model.head.load_state_dict(state["head"])
            if "residual_scale" in state:
                model.residual_scale.data.copy_(state["residual_scale"])
            print(f"  Loaded temporal transformer weights from {ckpt_path}")
        model.static_model.to(device)
        model.eval()
        return model, device

    model = TemporalEmotionModel().to(device)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(str(ckpt_path), map_location=device)
        if "transformer" in state:
            # Auto-detect transformer checkpoint
            return load_temporal_model(ckpt_path, device, is_transformer=True)
        model.gru.load_state_dict(state["gru"])
        model.attn.load_state_dict(state["attn"])
        model.head.load_state_dict(state["head"])
        if "residual_scale" in state:
            model.residual_scale.data.copy_(state["residual_scale"])
        if "backbone_last_block" in state:
            model.static_model.backbone.blocks[-1].load_state_dict(state["backbone_last_block"])
        print(f"  Loaded temporal weights from {ckpt_path}")
    model.static_model.to(device)
    model.eval()
    return model, device
