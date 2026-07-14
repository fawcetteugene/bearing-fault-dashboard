"""
Neural network model definitions.

Models defined here:
  FaultTransformer   — multi-head self-attention over feature tokens
  FaultCNN           — 1D convolutions treating features as a sequence
  FaultLSTM          — bidirectional LSTM over feature sequence
  MetaLearnerBase    — shared backbone for MAML and Meta-SGD
  FBCLModel          — continual learning model with expandable task heads

All models accept (batch, n_features) input and output (batch, n_classes) logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    TRANS_D_MODEL, TRANS_N_HEADS, TRANS_N_LAYERS, TRANS_DROPOUT,
    CNN_CHANNELS, CNN_DROPOUT,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    RESNET_CHANNELS, RESNET_DROPOUT,
    N_CLASSES,
)


class FaultTransformer(nn.Module):
    """
    Treats each of the 19 input features as a token.
    Self-attention learns which features interact to distinguish fault types.
    Pre-LayerNorm (norm_first=True) keeps gradients stable through the stack.
    """

    def __init__(self, n_features: int, n_classes: int = N_CLASSES,
                 d_model: int = TRANS_D_MODEL, n_heads: int = TRANS_N_HEADS,
                 n_layers: int = TRANS_N_LAYERS, dropout: float = TRANS_DROPOUT):
        super().__init__()
        self.embed    = nn.Linear(1, d_model)
        self.pos      = nn.Parameter(torch.randn(1, n_features, d_model) * 0.01)
        self.drop_in  = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self._attention_weights = None   # populated during forward for heatmaps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        x = x.unsqueeze(-1)          # (B, F, 1)
        x = self.embed(x) + self.pos
        x = self.drop_in(x)
        x = self.encoder(x)
        x = self.norm_out(x)
        x = x.mean(dim=1)            # global average pool over feature tokens
        return self.head(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the average attention weight matrix across heads and layers
        for visualisation. Shape: (n_features, n_features).
        """
        self.eval()
        hooks, weights = [], []

        def hook_fn(module, inp, out):
            # TransformerEncoderLayer stores attn weights in out[1]
            if isinstance(out, tuple) and len(out) == 2:
                weights.append(out[1].detach())

        for layer in self.encoder.layers:
            hooks.append(layer.self_attn.register_forward_hook(hook_fn))

        with torch.no_grad():
            x_tok = x.unsqueeze(-1)
            x_tok = self.embed(x_tok) + self.pos
            # need_weights=True to get attention output
            for layer in self.encoder.layers:
                x_tok, w = layer.self_attn(x_tok, x_tok, x_tok,
                                           need_weights=True, average_attn_weights=True)
                weights.append(w.detach())
                x_tok = layer.norm1(x_tok)

        for h in hooks:
            h.remove()

        if weights:
            return torch.stack(weights).mean(0).squeeze(0)  # (F, F)
        return torch.zeros(x.size(1), x.size(1))


class FaultCNN(nn.Module):
    """
    1D CNN that treats the 19 features as a short time series.
    Three conv blocks with batch norm, then a dense classifier.
    """

    def __init__(self, n_features: int, n_classes: int = N_CLASSES,
                 channels: list = CNN_CHANNELS, dropout: float = CNN_DROPOUT):
        super().__init__()
        layers = []
        in_ch  = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

        # Global average pool removes the spatial dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → treat F as sequence length with 1 channel
        x = x.unsqueeze(1)                  # (B, 1, F)
        x = self.conv_stack(x)              # (B, C_last, F)
        x = self.pool(x).squeeze(-1)        # (B, C_last)
        return self.head(x)


class FaultLSTM(nn.Module):
    """
    Bidirectional LSTM over the 19-feature sequence.
    Each feature is fed as a time step with a scalar value.
    """

    def __init__(self, n_features: int, n_classes: int = N_CLASSES,
                 hidden: int = LSTM_HIDDEN, n_layers: int = LSTM_LAYERS,
                 dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → (B, F, 1) as sequence of scalar features
        x, _ = self.lstm(x.unsqueeze(-1))   # (B, F, hidden*2)
        x     = self.norm(x[:, -1, :])      # take last time step
        return self.head(x)


class _ResBlock(nn.Module):
    """1D residual block: two conv layers with a skip connection."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        return F.gelu(out + self.skip(x))


class FaultResNet1D(nn.Module):
    """
    1D ResNet treating the feature vector as a short sequence.
    Residual connections help gradient flow and allow deeper networks
    without degradation — typically outperforms the plain CNN.
    """

    def __init__(self, n_features: int, n_classes: int = N_CLASSES,
                 channels: list = RESNET_CHANNELS, dropout: float = RESNET_DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(_ResBlock(channels[i], channels[i + 1], dropout))
        self.blocks = nn.Sequential(*blocks)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B, 1, F)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class MetaLearnerBase(nn.Module):
    """
    Shared backbone for MAML and Meta-SGD.
    Lightweight transformer encoder; fast enough for many inner-loop steps on CPU.
    """

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        d = 32   # small for fast inner-loop adaptation
        self.embed  = nn.Linear(1, d)
        self.pos    = nn.Parameter(torch.randn(1, n_features, d) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=64,
            dropout=0.05, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head    = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1)) + self.pos
        x = self.encoder(x).mean(dim=1)
        return self.head(x)


class FBCLModel(nn.Module):
    """
    Feature Boosting Continual Learning model.

    Shared encoder with a growing list of task heads.
    When a new fault severity task arrives, a residual adapter block is added
    and knowledge distillation keeps the existing heads stable.
    """

    def __init__(self, n_features: int, initial_classes: int):
        super().__init__()
        d = TRANS_D_MODEL

        self.embed  = nn.Linear(1, d)
        self.pos    = nn.Parameter(torch.randn(1, n_features, d) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=TRANS_N_HEADS,
            dim_feedforward=d * 2, dropout=TRANS_DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(layer, num_layers=TRANS_N_LAYERS)
        self.norm_out = nn.LayerNorm(d)

        # Task heads — one Linear per incremental task; first one is initial
        self.task_heads = nn.ModuleList([nn.Linear(d, initial_classes)])

        # Adapter blocks — added as new tasks arrive for residual feature boosting
        self.adapters   = nn.ModuleList()

    def add_task(self, n_new_classes: int):
        """Call before training on a new set of fault classes."""
        d = TRANS_D_MODEL
        self.task_heads.append(nn.Linear(d, n_new_classes))
        self.adapters.append(nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, d)
        ))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1)) + self.pos
        x = self.encoder(x)
        x = self.norm_out(x).mean(dim=1)
        return x

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        feat = self.encode(x)
        if task_id > 0 and (task_id - 1) < len(self.adapters):
            feat = feat + self.adapters[task_id - 1](feat)
        return self.task_heads[task_id](feat)


# ---------------------------------------------------------------------------
# Checkpoint-native models — match the exact architectures saved on disk.
# These are NOT used for training; they exist solely so artifacts.py can
# reconstruct the saved weights without key-name mismatches.
# ---------------------------------------------------------------------------

def _cls_head(d_in: int, hidden: int, n_classes: int) -> nn.Sequential:
    """Linear → BN → GELU → Linear → BN → GELU → Linear (3-layer head)."""
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.BatchNorm1d(hidden),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden // 2),
        nn.BatchNorm1d(hidden // 2),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden // 2, n_classes),
    )


class _CheckpointTransformer(nn.Module):
    """
    Generic transformer backbone matching the checkpoint key layout:
      pos_embedding, feature_embedding, transformer, layer_norm.
    Used by MAMLCheckpointModel, MetaSGDCheckpointModel, FBCLCheckpointModel.
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float, use_cls: bool = False,
                 dim_feedforward: int | None = None):
        super().__init__()
        self.use_cls = use_cls
        seq_len = n_features + (1 if use_cls else 0)
        ff = dim_feedforward if dim_feedforward is not None else d_model * 4

        self.pos_embedding    = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.feature_embedding = nn.Linear(1, d_model)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.layer_norm  = nn.LayerNorm(d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        tok = self.feature_embedding(x.unsqueeze(-1))   # (B, F, d)
        if self.use_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        tok = tok + self.pos_embedding
        tok = self.transformer(tok)
        tok = self.layer_norm(tok)
        return tok[:, 0] if self.use_cls else tok.mean(dim=1)


class MAMLCheckpointModel(_CheckpointTransformer):
    """
    Matches maml_final_model.pth:
      d=128, n_heads=4, n_layers=4, dim_feedforward=1024, use_cls=True,
      mask_token, reconstruction_head, classification_head (4-layer BN head).
    """

    def __init__(self, n_features: int = 19, n_classes: int = 10):
        super().__init__(n_features, d_model=128, n_heads=4, n_layers=4,
                         dropout=0.3, use_cls=True, dim_feedforward=1024)
        d = 128
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification_head(self.encode(x))


class MetaSGDCheckpointModel(_CheckpointTransformer):
    """
    Matches meta_sgd_model_improved.pth:
      d=320, n_heads=8 (320/8=40 — but checkpoint uses 8 heads on d=320),
      n_layers=8, no cls token, classification_head (3-layer BN head).
    """

    def __init__(self, n_features: int = 19, n_classes: int = 10):
        # 320 / 8 = 40 — valid head dim
        super().__init__(n_features, d_model=320, n_heads=8, n_layers=8,
                         dropout=0.1, use_cls=False)
        self.classification_head = nn.Sequential(
            nn.Linear(320, 640),
            nn.BatchNorm1d(640),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(320, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classification_head(self.encode(x))


class FBCLCheckpointModel(nn.Module):
    """
    Matches fbcl_model.pth:
      pos_embedding(1,19,256), feature_embedding Sequential(Linear→GELU→Linear),
      transformer (6 layers, d=256, n_heads=16), layer_norm,
      classifier (4-layer BN head), boosting_heads (ModuleList of 2 heads).
    Forward uses classifier + mean of boosting_head logits.
    """

    def __init__(self, n_features: int = 19, n_classes: int = 10):
        super().__init__()
        d = 256
        self.pos_embedding    = nn.Parameter(torch.randn(1, n_features, d) * 0.02)
        self.pos_scale        = nn.Parameter(torch.ones(1))
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, 128), nn.GELU(), nn.Linear(128, d)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=16, dim_feedforward=d * 4,
            dropout=0.15, activation="gelu",
            batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)
        self.layer_norm  = nn.LayerNorm(d)

        self.classifier = nn.Sequential(
            nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        self.boosting_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )
            for _ in range(2)
        ])

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.feature_embedding(x.unsqueeze(-1))
        tok = tok + self.pos_embedding * self.pos_scale
        tok = self.transformer(tok)
        return self.layer_norm(tok).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._encode(x)
        logits = self.classifier(feat)
        for head in self.boosting_heads:
            logits = logits + head(feat)
        return logits
