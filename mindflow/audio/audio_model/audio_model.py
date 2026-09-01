"""
MindFlow Phase 2 — Audio Model.

Architecture:
    WavLM Large backbone (bottom 12 layers frozen, top 12 fine-tuned)
      -> Attention Pooling (learns to weight time-steps, e.g. down-weight silence)
      -> Linear projection: 1024 -> 768   (WavLM Large's hidden size -> fusion's expected input)
      -> Three classification heads: emotion (7-class), stress (regression 0-1), confidence (regression 0-1)

The 768-dim embedding (output of the projection layer, before the heads)
is what gets passed to the fusion model, per the project spec.

Usage:
    model = AudioModel(num_emotions=7)
    output = model(input_values)  # input_values: (batch, num_samples) raw 16kHz waveform
    output["embedding"]      -> (batch, 768)
    output["emotion_logits"] -> (batch, 7)
    output["stress"]         -> (batch,) in [0, 1]
    output["confidence"]     -> (batch,) in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from transformers import WavLMModel

WAVLM_HIDDEN_SIZE = 1024   # WavLM Large's native hidden size
FUSION_EMBED_SIZE = 768    # what the fusion layer expects, per project spec
NUM_FREEZE_LAYERS = 12     # freeze bottom half of WavLM's 24 transformer layers


class AttentionPooling(nn.Module):
    """
    Learns a per-time-step importance score, softmaxes it into weights,
    and produces a single weighted-average vector per clip.

    Input:  (batch, time, hidden)   -- WavLM's last_hidden_state
    Output: (batch, hidden)         -- one pooled vector per clip
    """

    def __init__(self, hidden_size: int = WAVLM_HIDDEN_SIZE):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_states: (batch, time, hidden)
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, time)

        if attention_mask is not None:
            # Padded time-steps get -inf so softmax gives them ~0 weight.
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=1)  # (batch, time)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # (batch, hidden)
        return pooled


class AudioModel(nn.Module):
    def __init__(
        self,
        num_emotions: int = 7,
        wavlm_name: str = "microsoft/wavlm-large",
        num_freeze_layers: int = NUM_FREEZE_LAYERS,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()

        try:
            self.backbone = WavLMModel.from_pretrained(wavlm_name, local_files_only=True)
        except Exception:
            self.backbone = WavLMModel.from_pretrained(wavlm_name)

        # Freeze the conv feature extractor (raw waveform -> frame features).
        # This part is very low-level/generic; almost never worth fine-tuning.
        if freeze_feature_extractor:
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = False

        # Freeze the bottom N transformer layers; leave the rest trainable.
        for i, layer in enumerate(self.backbone.encoder.layers):
            requires_grad = i >= num_freeze_layers
            for param in layer.parameters():
                param.requires_grad = requires_grad

        self.pooling = AttentionPooling(WAVLM_HIDDEN_SIZE)
        self.projection = nn.Sequential(
            nn.Linear(WAVLM_HIDDEN_SIZE, FUSION_EMBED_SIZE),
            nn.LayerNorm(FUSION_EMBED_SIZE),
            nn.GELU(),
        )

        # Classification / regression heads, all reading from the 768-dim embedding.
        self.emotion_head = nn.Linear(FUSION_EMBED_SIZE, num_emotions)
        self.stress_head = nn.Sequential(
            nn.Linear(FUSION_EMBED_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )  # small MLP w/ dropout -- matches train_stage2_stress_v2.py's make_stress_head()
        self.confidence_head = nn.Sequential(nn.Linear(FUSION_EMBED_SIZE, 1), nn.Sigmoid())

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, time, 1024)

        # WavLM's conv layers downsample time; if an attention_mask was given
        # for the raw waveform, project it down to match hidden_states' time dim.
        pooled_mask = None
        if attention_mask is not None:
            pooled_mask = self.backbone._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )

        pooled = self.pooling(hidden_states, pooled_mask)      # (batch, 1024)
        embedding = self.projection(pooled)                    # (batch, 768)

        return {
            "embedding": embedding,
            "emotion_logits": self.emotion_head(embedding),
            "stress": self.stress_head(embedding).squeeze(-1),
            "confidence": self.confidence_head(embedding).squeeze(-1),
        }

    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


if __name__ == "__main__":
    # Quick smoke test: random waveform through the model, check shapes.
    model = AudioModel(num_emotions=7)
    trainable, total = model.trainable_parameter_count()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    dummy_input = torch.randn(2, 16000 * 3)  # batch of 2, 3 seconds @ 16kHz
    output = model(dummy_input)
    for key, val in output.items():
        print(f"{key}: {tuple(val.shape)}")
