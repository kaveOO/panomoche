"""Image-quality scorer shared by ``predict`` and ``serve``: TOPIQ-NR (see ``iqa.py``).

A scorer is a callable turning a prepared picture (see ``imaging.prepare``) into a prediction
dict with at least ``issues``, ``low_quality``, ``quality_score`` (higher is better),
``scores`` and ``model``. ``decision.py`` then judges the score against the sequence.
"""

from __future__ import annotations

from collections.abc import Callable

Scorer = Callable[[object], dict]


def make_scorer(
    abs_threshold: float = 0.32, abs_threshold_flat: float = 0.40, threads: int | None = None
) -> tuple[Scorer, dict]:
    """Return ``(scorer, info)``; ``info`` describes the model for UIs and logs."""
    import torch

    from .iqa import LOW_QUALITY, TopiqPredictor
    from .sequence import SEQUENCE_ISSUE

    if threads:
        torch.set_num_threads(threads)
    predictor = TopiqPredictor(abs_threshold, abs_threshold_flat=abs_threshold_flat)
    return predictor, {
        "scorer": "topiq",
        "model": predictor.model_id,
        "issues": [LOW_QUALITY, SEQUENCE_ISSUE],
        "thresholds": {LOW_QUALITY: abs_threshold, "low_quality_flat": abs_threshold_flat},
    }
