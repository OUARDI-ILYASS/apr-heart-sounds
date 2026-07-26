"""2-D CNN over log-Mel spectrograms, with Grad-CAM-ready hooks.

Architecture (default): four ``Conv-BN-ReLU-MaxPool`` blocks with channel
widths 16/32/64/128, global average pooling, one hidden FC layer, two logits.
Roughly 0.2 M parameters.

Input  : (batch, 1, 32 mel bands, 188 frames)
Block 1: (batch, 16, 16, 94)
Block 2: (batch, 32,  8, 47)
Block 3: (batch, 64,  4, 23)
Block 4: (batch,128,  2, 11)   <- Grad-CAM target
GAP    : (batch,128)
FC     : (batch, 64) -> (batch, 2)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool.

    BatchNorm before ReLU is the original ordering and is what almost all
    audio-CNN baselines use; it also stabilises training at the small batch
    sizes we can afford.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, pool_size: int = 2, batch_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=not batch_norm)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return self.pool(x)


class HeartSoundCNN(nn.Module):
    """Compact 2-D CNN for PCG log-Mel classification."""

    def __init__(self, n_mels: int = 32, n_frames: int = 188,
                 conv_channels: Tuple[int, ...] = (16, 32, 64, 128),
                 kernel_size: int = 3, padding: int = 1, pool_size: int = 2,
                 batch_norm: bool = True, dropout: float = 0.3,
                 fc_hidden: int = 64, n_classes: int = 2,
                 global_pool: str = "avg"):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.conv_channels = tuple(conv_channels)
        self.global_pool = global_pool

        blocks: List[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            blocks.append(ConvBlock(in_channels, out_channels, kernel_size,
                                    padding, pool_size, batch_norm))
            in_channels = out_channels
        # Named ``blocks`` so Grad-CAM can address ``model.blocks[-1]``
        # without string surgery.
        self.blocks = nn.ModuleList(blocks)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_channels[-1], fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_classes)

        # Populated by Grad-CAM via forward/backward hooks.
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                     # (B, mels, frames) -> add channel
            x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        # Keep a handle on the last conv feature map for Grad-CAM. Storing it
        # here (rather than via an external hook) means the tensor is always
        # the *pre-pooling* map, which is what Grad-CAM needs.
        if x.requires_grad:
            self._activations = x
            x.register_hook(self._save_gradients)

        if self.global_pool == "avg":
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.global_pool == "max":
            x = F.adaptive_max_pool2d(x, 1)
        else:
            raise ValueError(f"Unknown global_pool: {self.global_pool}")

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)

    def _save_gradients(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    # ------------------------------------------------------------------ #
    @property
    def last_conv_layer(self) -> nn.Module:
        """The Grad-CAM target layer."""
        return self.blocks[-1].conv

    def feature_map_shape(self) -> Tuple[int, int, int]:
        """(channels, freq, time) of the last conv map, computed analytically."""
        pool_factor = 2 ** len(self.blocks)
        return (self.conv_channels[-1],
                self.n_mels // pool_factor,
                self.n_frames // pool_factor)

    def n_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        conv = sum(p.numel() for b in self.blocks for p in b.parameters())
        return {"total": total, "trainable": trainable,
                "conv": conv, "head": total - conv}

    def architecture_spec(self) -> Dict[str, object]:
        """Everything needed to rebuild this model from a checkpoint."""
        return {
            "n_mels": self.n_mels, "n_frames": self.n_frames,
            "conv_channels": list(self.conv_channels),
            "global_pool": self.global_pool,
            "fc_hidden": self.fc1.out_features,
            "n_classes": self.fc2.out_features,
            "dropout": float(self.dropout.p),
            "feature_map_shape": list(self.feature_map_shape()),
            "n_parameters": self.n_parameters(),
        }


def build_cnn(cfg: Dict, n_mels: int, n_frames: int) -> HeartSoundCNN:
    """Instantiate the CNN from the config."""
    arch = cfg["cnn"]["architecture"]
    return HeartSoundCNN(
        n_mels=n_mels,
        n_frames=n_frames,
        conv_channels=tuple(arch["conv_channels"]),
        kernel_size=int(arch["kernel_size"]),
        padding=int(arch["padding"]),
        pool_size=int(arch["pool_size"]),
        batch_norm=bool(arch["batch_norm"]),
        dropout=float(arch["dropout"]),
        fc_hidden=int(arch["fc_hidden"]),
        n_classes=int(arch["n_classes"]),
        global_pool=str(arch.get("global_pool", "avg")),
    )


def rebuild_from_checkpoint(checkpoint: Dict) -> HeartSoundCNN:
    """Reconstruct a model from a checkpoint's stored architecture spec.

    This is why we save the spec alongside the weights: a bare ``state_dict``
    cannot tell you how many channels block 3 had, and guessing wrong produces
    a shape error that looks like data corruption.
    """
    spec = checkpoint["architecture"]
    model = HeartSoundCNN(
        n_mels=spec["n_mels"], n_frames=spec["n_frames"],
        conv_channels=tuple(spec["conv_channels"]),
        dropout=spec.get("dropout", 0.3),
        fc_hidden=spec.get("fc_hidden", 64),
        n_classes=spec.get("n_classes", 2),
        global_pool=spec.get("global_pool", "avg"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def resolve_device(requested: str = "cuda") -> torch.device:
    """Honour the configured device, falling back gracefully.

    Prints nothing - the caller logs the outcome so it lands in the phase
    summary and the paper's reproducibility section.
    """
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")
