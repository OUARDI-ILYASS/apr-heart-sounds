"""Grad-CAM for the log-Mel CNN.

Grad-CAM (Selvaraju et al., 2017) answers: which regions of the input did the
network use to produce *this* class score? It works by taking the gradient of
the class logit with respect to the last convolutional feature map, average-
pooling those gradients per channel to get a channel importance weight, and
summing the channels with those weights.

    alpha_k = GAP( d y_c / d A_k )          # importance of channel k
    L = ReLU( sum_k alpha_k * A_k )         # class-discriminative map

In our case the resulting map lives on a (2 frequency x 11 time) grid, which we
upsample back to (32 x 188) - the log-Mel input resolution.

PROFESSOR Q: "Why ReLU at the end?"
A: Because we want the evidence *for* the class, not against it. Negative
   values in the weighted sum are regions whose activation pushes the score
   down; including them would blur "where the murmur is" with "where the
   absence of a murmur is". The ReLU is what makes the map class-discriminative
   rather than just class-sensitive.

PROFESSOR Q: "Grad-CAM on a 2x11 grid is very coarse. Is that a problem?"
A: It is a real limitation and we quantify it rather than hide it. On the
   frequency axis, 2 cells over 32 mel bands is too coarse to localise a murmur
   spectrally, so we deliberately make no frequency claims from Grad-CAM - the
   frequency-domain evidence comes from PWP SHAP, where the mapping to Hz is
   exact. On the time axis, 11 cells over 3 seconds is ~273 ms per cell, which
   is comparable to the duration of systole itself (~300 ms). That is coarse
   but sufficient to distinguish "attends during systole" from "attends during
   diastole", which is the only temporal claim we make. We also report
   Grad-CAM++ and a Guided-Grad-CAM variant as robustness checks, and phase 08
   verifies the three agree.

PROFESSOR Q: "Isn't Grad-CAM known to be unreliable?"
A: There is a real literature on saliency-map failure modes - Adebayo et al.'s
   sanity checks showed that some methods produce plausible maps even from a
   randomly initialised network. We run exactly that test: ``sanity_check_
   randomization`` recomputes the maps from a re-initialised model and reports
   the correlation with the trained model's maps. A high correlation would mean
   our maps are an artefact of architecture rather than of learning, and would
   invalidate the XAI claim. That check is part of phase 08, not an optional
   extra.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM (and Grad-CAM++) for a CNN with a nominated conv layer."""

    def __init__(self, model, target_layer=None, upsample_mode: str = "bilinear"):
        self.model = model.eval()
        self.target_layer = target_layer if target_layer is not None else model.last_conv_layer
        self.upsample_mode = upsample_mode

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List = []
        self._register_hooks()

    # ------------------------------------------------------------------ #
    def _register_hooks(self) -> None:
        """Capture the target layer's output and the gradient flowing into it.

        We hook the *layer* rather than reading the tensor stored on the model,
        because hooks work identically for any layer, which is what makes the
        layer-sensitivity analysis below possible.
        """
        def forward_hook(module, inputs, output):
            self._activations = output
            # retain_grad is needed because this is a non-leaf tensor.
            output.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Always call this. Leaked hooks silently slow down later inference."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove_hooks()

    # ------------------------------------------------------------------ #
    def generate(self, x: torch.Tensor, class_index: Optional[int] = None,
                 method: str = "gradcam", normalize: str = "unit_mass"
                 ) -> Dict[str, object]:
        """Compute the attribution map for one batch.

        Parameters
        ----------
        x
            Input tensor, shape (B, mels, frames) or (B, 1, mels, frames).
        class_index
            Which logit to explain. ``None`` explains the predicted class,
            which is the honest default: we want to know why the model said
            what it said, not why it might have said something else.
        method
            ``"gradcam"`` or ``"gradcampp"``.
        normalize
            ``"unit_mass"`` (map sums to 1) or ``"minmax"`` (map in [0, 1]).
            Unit mass is the right choice for us because the alignment metric
            is a *fraction of attribution mass*, which only makes sense if the
            total mass is fixed across examples.
        """
        self.model.zero_grad(set_to_none=True)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.clone().requires_grad_(True)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        if class_index is None:
            targets = logits.argmax(dim=1)
        else:
            targets = torch.full((x.size(0),), int(class_index),
                                 dtype=torch.long, device=x.device)

        selected = logits.gather(1, targets.unsqueeze(1)).squeeze(1).sum()
        selected.backward()

        activations = self._activations           # (B, C, Hf, Wf)
        gradients = self._gradients               # (B, C, Hf, Wf)
        if activations is None or gradients is None:
            raise RuntimeError(
                "Hooks captured nothing. The target layer is probably not on "
                "the forward path of this model."
            )

        if method == "gradcam":
            # Channel weight = spatial average of the gradient.
            weights = gradients.mean(dim=(2, 3), keepdim=True)
        elif method == "gradcampp":
            # Grad-CAM++ weights each spatial position by a second-order term,
            # which handles multiple disjoint evidence regions better - relevant
            # here because a murmur recurs in every cardiac cycle, so the
            # evidence is genuinely multi-modal in time.
            grad2 = gradients ** 2
            grad3 = grad2 * gradients
            sum_activations = activations.sum(dim=(2, 3), keepdim=True)
            denominator = 2.0 * grad2 + sum_activations * grad3
            denominator = torch.where(denominator != 0, denominator,
                                      torch.ones_like(denominator))
            alpha = grad2 / denominator
            weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Unknown Grad-CAM method: {method}")

        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode=self.upsample_mode, align_corners=False)
        cam = cam.squeeze(1)                       # (B, mels, frames)
        cam_np = cam.detach().cpu().numpy()
        cam_np = _normalize_maps(cam_np, normalize)

        return {
            "cam": cam_np,
            "predicted_class": targets.detach().cpu().numpy(),
            "probabilities": probs.detach().cpu().numpy(),
            "feature_map_shape": list(activations.shape[1:]),
            "method": method,
            "normalize": normalize,
        }


def _normalize_maps(cam: np.ndarray, mode: str) -> np.ndarray:
    """Normalise each map in a batch independently."""
    out = np.asarray(cam, dtype=np.float64).copy()
    for i in range(len(out)):
        single = out[i]
        if mode == "unit_mass":
            total = single.sum()
            out[i] = single / total if total > 0 else np.full_like(single, 1.0 / single.size)
        elif mode == "minmax":
            span = single.max() - single.min()
            out[i] = (single - single.min()) / span if span > 0 else np.zeros_like(single)
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalisation mode: {mode}")
    return out


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def compute_gradcam_batch(model, X: np.ndarray, device, batch_size: int = 32,
                          method: str = "gradcam", class_index: Optional[int] = None,
                          normalize: str = "unit_mass",
                          progress: bool = False) -> Dict[str, np.ndarray]:
    """Grad-CAM over a whole array of segments."""
    model = model.to(device).eval()
    cams: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []

    with GradCAM(model) as explainer:
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(
                np.asarray(X[start:start + batch_size], dtype=np.float32)
            ).to(device)
            result = explainer.generate(batch, class_index=class_index,
                                        method=method, normalize=normalize)
            cams.append(result["cam"])
            predictions.append(result["predicted_class"])
            probabilities.append(result["probabilities"])
            if progress and (start // batch_size) % 10 == 0:
                print(f"\r  grad-cam: {min(start + batch_size, len(X))}/{len(X)}",
                      end="", flush=True)
    if progress:
        print()

    return {
        "cams": np.concatenate(cams, axis=0),
        "predicted_class": np.concatenate(predictions),
        "probabilities": np.concatenate(probabilities, axis=0),
    }


# --------------------------------------------------------------------------- #
# Aggregation and sanity checks
# --------------------------------------------------------------------------- #
def average_cam_by_class(cams: np.ndarray, labels: np.ndarray,
                         predictions: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean attribution map per outcome category.

    Averaging over many examples suppresses per-example noise and reveals the
    systematic pattern - which is what supports a population-level claim like
    "the model attends to systole". A single striking heatmap is an anecdote;
    an average over hundreds of correctly classified abnormals is evidence.
    """
    cams = np.asarray(cams)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    categories = {
        "true_positive": (labels == 1) & (predictions == 1),
        "true_negative": (labels == 0) & (predictions == 0),
        "false_positive": (labels == 0) & (predictions == 1),
        "false_negative": (labels == 1) & (predictions == 0),
        "all_abnormal": labels == 1,
        "all_normal": labels == 0,
    }

    out: Dict[str, np.ndarray] = {}
    for name, mask in categories.items():
        out[name] = cams[mask].mean(axis=0) if mask.sum() else np.zeros(cams.shape[1:])
        out[f"{name}_count"] = int(mask.sum())
    return out


def cam_frequency_profile(cam: np.ndarray, band_frequencies: np.ndarray
                          ) -> Dict[str, object]:
    """Marginalise a map over time to get attribution vs frequency.

    Reported with the caveat that the frequency axis of Grad-CAM has only
    ``n_mels / 2^n_blocks`` genuinely independent cells (2 by default), so this
    profile is heavily smoothed by the upsampling and should not be read as
    fine-grained spectral evidence.
    """
    profile = np.asarray(cam).sum(axis=1)
    total = profile.sum()
    normalized = profile / total if total > 0 else profile
    return {
        "profile": normalized.tolist(),
        "frequencies_hz": np.asarray(band_frequencies).tolist(),
        "peak_frequency_hz": float(band_frequencies[int(np.argmax(normalized))]),
        "centroid_hz": float(np.sum(normalized * band_frequencies)),
        "effective_resolution_note": "Upsampled from a coarse feature map; "
                                     "treat as a trend, not as spectral detail.",
    }


def cam_time_profile(cam: np.ndarray, frame_times: np.ndarray) -> Dict[str, object]:
    """Marginalise a map over frequency to get attribution vs time."""
    profile = np.asarray(cam).sum(axis=0)
    total = profile.sum()
    normalized = profile / total if total > 0 else profile
    return {
        "profile": normalized.tolist(),
        "times_s": np.asarray(frame_times).tolist(),
        "peak_time_s": float(frame_times[int(np.argmax(normalized))]),
        # Temporal concentration: 0 = uniform over time, 1 = a single frame.
        "concentration": float(1.0 - (-(normalized[normalized > 0] *
                                        np.log(normalized[normalized > 0])).sum()
                                      / np.log(len(normalized)))),
    }


def sanity_check_randomization(model_class, checkpoint: Dict, X: np.ndarray,
                               device, n_samples: int = 100, seed: int = 42
                               ) -> Dict[str, float]:
    """Adebayo et al. model-randomisation sanity check.

    Recompute Grad-CAM from a randomly re-initialised copy of the network and
    correlate the maps with the trained model's. If a saliency method is
    genuinely reading learned structure, that correlation should be low. A high
    correlation means the maps are determined by architecture and input
    statistics rather than by anything the model learned - in which case no
    conclusion may be drawn from them.

    We report the number. If it comes out high, that goes in the limitations
    section, not in the bin.
    """
    from ..models.cnn import rebuild_from_checkpoint

    subset = X[:n_samples]

    trained = rebuild_from_checkpoint(checkpoint).to(device)
    trained_cams = compute_gradcam_batch(trained, subset, device)["cams"]

    random_model = rebuild_from_checkpoint(checkpoint).to(device)
    torch.manual_seed(seed)
    for module in random_model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    random_cams = compute_gradcam_batch(random_model, subset, device)["cams"]

    correlations = []
    for trained_map, random_map in zip(trained_cams, random_cams):
        a, b = trained_map.ravel(), random_map.ravel()
        if a.std() > 1e-12 and b.std() > 1e-12:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))

    correlations = np.asarray(correlations) if correlations else np.array([np.nan])
    mean_correlation = float(np.nanmean(correlations))
    return {
        "mean_correlation_with_random_model": mean_correlation,
        "median_correlation": float(np.nanmedian(correlations)),
        "n_compared": int(len(correlations)),
        "passes_sanity_check": bool(abs(mean_correlation) < 0.3),
        "interpretation": (
            "Low correlation: Grad-CAM maps depend on learned weights, as required."
            if abs(mean_correlation) < 0.3 else
            "HIGH correlation: maps are largely architecture-driven. Any claim "
            "based on them must be withdrawn or heavily qualified."
        ),
    }


def layer_sensitivity(model, X: np.ndarray, device, n_samples: int = 50
                      ) -> List[Dict[str, object]]:
    """How much does the choice of target layer change the map?

    Reported as a robustness check for the "why the last conv layer?" question.
    We expect earlier layers to give higher-resolution but less
    class-discriminative maps; if the *temporal* conclusion (attention in
    systole) survives across layers, it is not an artefact of layer choice.
    """
    subset = X[:n_samples]
    rows: List[Dict[str, object]] = []
    reference: Optional[np.ndarray] = None

    for index, block in enumerate(model.blocks):
        with GradCAM(model, target_layer=block.conv) as explainer:
            batch = torch.from_numpy(np.asarray(subset, dtype=np.float32)).to(device)
            cams = explainer.generate(batch)["cam"]

        time_profile = cams.sum(axis=1)
        time_profile = time_profile / (time_profile.sum(axis=1, keepdims=True) + 1e-12)

        row: Dict[str, object] = {
            "layer": f"conv_block_{index + 1}",
            "mean_temporal_concentration": float(np.mean([
                1.0 + (p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p))
                for p in time_profile
            ])),
        }
        if reference is None:
            reference = time_profile
            row["correlation_with_last"] = None
        else:
            correlations = [
                float(np.corrcoef(a, b)[0, 1])
                for a, b in zip(time_profile, reference)
                if a.std() > 1e-12 and b.std() > 1e-12
            ]
            row["correlation_with_last"] = float(np.mean(correlations)) if correlations else None
        rows.append(row)

    return rows
