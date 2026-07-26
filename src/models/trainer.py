"""CNN training: dataset, augmentation, loop, early stopping, checkpointing.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..utils.io import save_checkpoint
from ..utils.seed import worker_init_fn


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class LogMelDataset(Dataset):
    """In-memory dataset of log-Mel segments.

    The whole training set is roughly 25k x 32 x 188 float32 = ~600 MB, which
    fits comfortably in RAM. Keeping it in memory removes disk I/O from the
    training loop entirely, which matters because these segments are small and
    the GPU would otherwise spend most of its time waiting.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 groups: Optional[np.ndarray] = None,
                 augment: bool = False, augment_cfg: Optional[Dict] = None,
                 seed: int = 42):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.groups = groups
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self._rng = np.random.default_rng(seed)

        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} rows but y has {len(self.y)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[index].copy()
        if self.augment:
            x = self._apply_augmentation(x)
        return torch.from_numpy(x), torch.tensor(self.y[index])

    def _apply_augmentation(self, x: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg

        # Circular time shift. Physiologically motivated: the phase at which a
        # 3 s window happens to start is arbitrary, so the label must be
        # invariant to it. This is the augmentation most likely to help.
        if cfg.get("time_shift", False):
            max_shift = int(cfg.get("time_shift_max_frac", 0.2) * x.shape[1])
            if max_shift > 0:
                x = np.roll(x, int(self._rng.integers(-max_shift, max_shift + 1)), axis=1)

        # Additive Gaussian noise in the log-Mel domain - a crude stand-in for
        # variation in the recording noise floor across sites.
        if cfg.get("gaussian_noise", False):
            x = x + self._rng.normal(0.0, float(cfg.get("noise_std", 0.01)), x.shape)

        # Deliberately NOT applying SpecAugment frequency masking: masking a
        # band can delete the murmur itself, turning an abnormal example into a
        # mislabelled normal one, and it would also distort the frequency-axis
        # Grad-CAM analysis in phase 08.
        return x.astype(np.float32)


def make_dataloaders(X_train, y_train, X_val, y_val, cfg: Dict,
                     seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    train_cfg = cfg["cnn"]["training"]
    aug_cfg = cfg["cnn"].get("augmentation", {})

    train_ds = LogMelDataset(
        X_train, y_train, augment=bool(aug_cfg.get("enabled", False)),
        augment_cfg=aug_cfg, seed=seed,
    )
    val_ds = LogMelDataset(X_val, y_val, augment=False)

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": bool(train_cfg.get("pin_memory", False)),
    }
    if common["num_workers"] > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=False,
        worker_init_fn=worker_init_fn, generator=generator, **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def compute_class_weights(y: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1.

    Normalising to mean 1 keeps the loss magnitude comparable to the unweighted
    case, so the learning rate does not need retuning when weighting is toggled
    (which is exactly what ablation A3 does).
    """
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Metrics used inside the loop
# --------------------------------------------------------------------------- #
def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def quick_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy plus MAcc - enough to drive early stopping without sklearn."""
    tp, tn, fp, fn = _binary_counts(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "macc": 0.5 * (sensitivity + specificity),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Patience-based early stopping on a maximised metric."""

    def __init__(self, patience: int = 12, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = -np.inf
        self.counter: int = 0
        self.best_epoch: int = -1

    def step(self, value: float, epoch: int) -> bool:
        """Return True if training should stop."""
        if value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, use_amp: bool = False) -> Dict[str, float]:
    model.train()
    total_loss, n_seen = 0.0, 0
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            # Unscale before clipping so the clip threshold means what it says.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.detach()) * xb.size(0)
        n_seen += xb.size(0)
        preds.append(logits.detach().argmax(1).cpu().numpy())
        targets.append(yb.detach().cpu().numpy())

    metrics = quick_metrics(np.concatenate(targets), np.concatenate(preds))
    metrics["loss"] = total_loss / max(1, n_seen)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, Any]:
    model.eval()
    total_loss, n_seen = 0.0, 0
    probs: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += float(loss) * xb.size(0)
        n_seen += xb.size(0)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(yb.cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    metrics = quick_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, n_seen)
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred,
            "y_prob": np.concatenate(probs)}


def train_model(model, train_loader, val_loader, cfg: Dict, device,
                class_weights: Optional[torch.Tensor] = None,
                checkpoint_path: Optional[str | Path] = None,
                logger=None) -> Dict[str, Any]:
    """Full training loop. Returns history plus the best epoch's state dict."""
    train_cfg = cfg["cnn"]["training"]
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=float(train_cfg["learning_rate"]),
            momentum=0.9, weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = None
    if str(train_cfg.get("scheduler", "")).lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 5)),
        )

    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    monitor = str(train_cfg.get("monitor", "val_macc")).replace("val_", "")
    stopper = EarlyStopping(patience=int(train_cfg.get("early_stopping_patience", 12)))

    history: List[Dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: Dict[str, float] = {}
    epochs = int(train_cfg["epochs"])
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer,
                                        device, amp_scaler, use_amp)
        val_out = evaluate(model, val_loader, criterion, device)
        val_metrics = val_out["metrics"]

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_metrics[monitor])

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "seconds": round(time.perf_counter() - epoch_start, 2),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if logger is not None:
            logger.info(
                f"epoch {epoch:3d}/{epochs} | "
                f"train loss {train_metrics['loss']:.4f} macc {train_metrics['macc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} macc {val_metrics['macc']:.4f} "
                f"se {val_metrics['sensitivity']:.3f} sp {val_metrics['specificity']:.3f} | "
                f"lr {current_lr:.2e} | {record['seconds']:.1f}s"
            )

        # Track the best epoch by the monitored metric, keeping its weights.
        if val_metrics[monitor] > stopper.best:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)

        if stopper.step(val_metrics[monitor], epoch):
            if logger is not None:
                logger.info(
                    f"Early stopping at epoch {epoch}; best val_{monitor}="
                    f"{stopper.best:.4f} at epoch {stopper.best_epoch}"
                )
            break

    # Restore the best weights - NOT the last epoch's.
    model.load_state_dict(best_state)

    result: Dict[str, Any] = {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_val_metrics": best_metrics,
        f"best_val_{monitor}": stopper.best,
        "epochs_run": len(history),
        "early_stopped": len(history) < epochs,
        "training_seconds": round(time.perf_counter() - start, 1),
        "used_amp": use_amp,
        "device": str(device),
    }

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, {
            "state_dict": best_state,
            "architecture": model.architecture_spec(),
            "config_hash": cfg.get("_config_hash"),
            "best_val_metrics": best_metrics,
            "best_epoch": stopper.best_epoch,
            "history": history,
            "class_weights": class_weights.tolist() if class_weights is not None else None,
        })
        result["checkpoint"] = str(checkpoint_path)

    return result


def diagnose_training(history: List[Dict[str, Any]]) -> List[str]:
    """Turn training curves into plain-language warnings.

    These land in the phase summary so that "did it overfit?" is answered by an
    artifact rather than by squinting at a loss curve.
    """
    warnings: List[str] = []
    if not history:
        return ["No training history recorded."]

    last = history[-1]
    gap = last.get("train_macc", 0) - last.get("val_macc", 0)
    if gap > 0.15:
        warnings.append(
            f"Train-val MAcc gap is {gap:.3f} at the final epoch - the CNN is "
            "overfitting. Consider more dropout, stronger augmentation, or fewer blocks."
        )
    if last.get("val_macc", 0) < 0.6:
        warnings.append(
            f"Final validation MAcc is only {last.get('val_macc', 0):.3f}, close to "
            "chance (0.5). Check class weighting and the learning rate."
        )
    if last.get("val_sensitivity", 1) < 0.4:
        warnings.append(
            f"Sensitivity is {last.get('val_sensitivity', 0):.3f}: the model is missing "
            "most abnormal cases. In a screening context this is the expensive error."
        )
    val_losses = [h["val_loss"] for h in history]
    if len(val_losses) > 10 and val_losses[-1] > min(val_losses) * 1.2:
        warnings.append(
            "Validation loss has risen more than 20% above its minimum - classic "
            "overfitting. Early stopping restored the best weights, but the gap "
            "should be reported."
        )
    return warnings
