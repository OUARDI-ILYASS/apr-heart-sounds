"""Fixed-window segmentation of preprocessed recordings.

PROFESSOR Q: "Why fixed windows rather than segmenting into cardiac cycles?"
A: Two reasons, one practical and one methodological.
   Practical: cycle-based segmentation requires a segmenter, and a segmenter
   that fails on noisy recordings would silently bias which recordings survive.
   Methodological: our XAI claim is that the CNN discovers the systolic window
   *without being told where it is*. If we had already cut the signal at cycle
   boundaries, that discovery would be built into the input representation and
   the claim would be circular. Fixed windows keep the cardiac timing latent.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def segment_signal(x: np.ndarray, sr: float, segment_seconds: float,
                   hop_seconds: float, pad_policy: str = "drop"
                   ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Slice a signal into overlapping fixed-length windows.

    Returns ``(segments, metadata)`` where ``segments`` has shape
    ``(n_segments, segment_samples)`` and metadata records each window's
    position in the source recording - needed later to map an explanation back
    onto the original timeline.
    """
    seg_len = int(round(segment_seconds * sr))
    hop = int(round(hop_seconds * sr))
    if seg_len <= 0 or hop <= 0:
        raise ValueError("segment_seconds and hop_seconds must be positive")

    x = np.asarray(x, dtype=np.float32)
    segments: List[np.ndarray] = []
    meta: List[Dict[str, float]] = []

    if len(x) < seg_len:
        if pad_policy == "zero_pad":
            padded = np.zeros(seg_len, dtype=np.float32)
            padded[: len(x)] = x
            return padded[None, :], [{"index": 0, "start_sample": 0,
                                      "start_s": 0.0, "end_s": segment_seconds,
                                      "padded": True}]
        return np.empty((0, seg_len), dtype=np.float32), []

    n_full = 1 + (len(x) - seg_len) // hop
    for i in range(n_full):
        start = i * hop
        segments.append(x[start: start + seg_len])
        meta.append({
            "index": i,
            "start_sample": int(start),
            "start_s": round(start / sr, 4),
            "end_s": round((start + seg_len) / sr, 4),
            "padded": False,
        })

    # Trailing remainder
    consumed = (n_full - 1) * hop + seg_len
    if pad_policy == "zero_pad" and consumed < len(x):
        tail = x[consumed:]
        if len(tail) > 0.5 * seg_len:      # only keep a substantial remainder
            padded = np.zeros(seg_len, dtype=np.float32)
            padded[: len(tail)] = tail
            segments.append(padded)
            meta.append({
                "index": n_full, "start_sample": int(consumed),
                "start_s": round(consumed / sr, 4),
                "end_s": round((consumed + seg_len) / sr, 4), "padded": True,
            })

    return np.stack(segments).astype(np.float32), meta


def hop_for_split(split: str, cfg: Dict) -> float:
    """Overlapping windows on train, non-overlapping on val/test.

    Overlap on train is augmentation: it multiplies the number of CNN training
    examples without new data. Overlap on val/test would make the evaluation
    windows statistically dependent, which narrows confidence intervals without
    adding information - i.e. it would make the results look more certain than
    they are.
    """
    return float(cfg["train_hop_seconds"] if split == "train" else cfg["eval_hop_seconds"])
