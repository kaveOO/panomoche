"""Off-the-shelf no-reference image quality models (pyiqa), no training needed.

* ``topiq_nr-spaq`` and ``musiq-spaq``: overall quality, trained on SPAQ
  (11k smartphone photos), the closest public dataset to Panoramax pictures.
* ``liqe_mix``: CLIP-based; scores every (quality, scene, distortion) text
  triple, so marginalising gives a distortion type and a scene type as well.

Pictures are split into the same views as the DINOv3 pipeline (see
``imaging.py``), so 360° pictures are judged on their horizon band and a defect
in one direction shows up in the worst view.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import __version__

METRICS = ("topiq_nr-spaq", "musiq-spaq", "liqe_mix")
TOPIQ = "topiq_nr-spaq"
LOW_QUALITY = "low_quality"


def _liqe_joint(net, x: torch.Tensor) -> torch.Tensor:
    """Eval-mode LIQE forward returning the full (quality, scene, distortion) distribution.

    Same computation as ``pyiqa.archs.liqe_arch.LIQE.forward``, which only returns the quality marginal.
    """
    from pyiqa.archs.liqe_arch import dists_map, qualitys, scenes

    bs = x.size(0)
    x = (x - net.default_mean.to(x)) / net.default_std.to(x)
    x = x.unfold(2, 224, net.step).unfold(3, 224, net.step).permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, 3, 224, 224)
    num_patch = min(x.size(1), net.num_patch)
    sel = (torch.arange(num_patch) * max(1, x.size(1) // num_patch)).long()
    x = x[:, sel].reshape(bs * num_patch, 3, 224, 224)
    feats = net.clip_model.encode_image(x, pos_embedding=True)
    feats = feats / feats.norm(dim=1, keepdim=True)
    logits = net.clip_model.logit_scale.exp() * feats @ net.text_features.to(x).t()
    probs = F.softmax(logits.view(bs, num_patch, -1).mean(1), dim=1)
    return probs.view(bs, len(qualitys), len(scenes), len(dists_map))


def _chunked_text_features(self, x):
    """Replacement for ``LIQE.get_text_features``.

    The stock version encodes all 495 prompts in one batch with autograd on and
    peaks above 6 GB on first use, enough to get OOM-killed on an 8 GB machine.
    The result is cached by pyiqa, so this only matters the first time.
    """
    feats = []
    with torch.no_grad():
        for chunk in self.joint_texts.split(32):
            f = self.clip_model.encode_text(chunk.to(x.device))
            feats.append(f / f.norm(dim=1, keepdim=True))
    return torch.cat(feats)


class IQAScorer:
    def __init__(self, metrics=METRICS, device: str | None = None):
        import pyiqa
        from pyiqa.archs import liqe_arch

        liqe_arch.LIQE.get_text_features = _chunked_text_features
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = {m: pyiqa.create_metric(m, device=self.device) for m in metrics}

    @torch.inference_mode()
    def score_view(self, rgb: np.ndarray) -> dict:
        from pyiqa.archs.liqe_arch import dists_map, scenes

        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        out = {}
        for name, metric in self.metrics.items():
            if name == "liqe_mix":
                joint = _liqe_joint(metric.net, x)[0]
                q = joint.sum((1, 2))
                out[name] = float((q * torch.arange(1, 6, dtype=q.dtype)).sum())
                out["liqe_distortion"] = dict(zip(dists_map, joint.sum((0, 1)).tolist()))
                out["liqe_scene"] = dict(zip(scenes, joint.sum((0, 2)).tolist()))
            else:
                out[name] = float(metric(x).item())
        return out

    def score_picture(self, views: list[np.ndarray]) -> dict:
        """Per-metric mean and worst (min) over views; LIQE distributions averaged over views."""
        per_view = [self.score_view(v) for v in views]
        res: dict = {"views": len(views)}
        for name in self.metrics:
            vals = [v[name] for v in per_view]
            res[name] = float(np.mean(vals))
            res[f"{name}_min"] = float(np.min(vals))
            res[f"{name}_views"] = [round(v, 4) for v in vals]
        if "liqe_mix" in self.metrics:
            for key in ("liqe_distortion", "liqe_scene"):
                labels = per_view[0][key]
                res[key] = {k: float(np.mean([v[key][k] for v in per_view])) for k in labels}
        return res


class TopiqPredictor:
    """TOPIQ-NR (SPAQ) as a drop-in scorer for ``panomoche predict``.

    Flags ``low_quality`` when the worst view scores below a threshold that
    depends on the projection, because TOPIQ rates 360° pictures lower than
    flat photos (medians 0.57 and 0.73 on 700 random Panoramax pictures):

    * 360°: 0.32, which only catches clearly bad pictures;
    * flat: 0.40, which catches night, heavy fog and big obstructions.
      The final decision is made in ``decision.py``.
    """

    def __init__(self, abs_threshold: float = 0.32, scorer: IQAScorer | None = None, abs_threshold_flat: float = 0.40):
        self.abs_threshold = abs_threshold  # 360° pictures
        self.abs_threshold_flat = abs_threshold_flat
        self.scorer = scorer or IQAScorer(metrics=(TOPIQ,))
        self.model_id = f"panomoche-{TOPIQ}/{__version__}"

    def __call__(self, prep) -> dict:
        r = self.scorer.score_picture(prep.views())
        worst = round(r[f"{TOPIQ}_min"], 4)
        threshold = self.abs_threshold if prep.is_pano else self.abs_threshold_flat
        issues = [LOW_QUALITY] if worst < threshold else []
        return {
            "issues": issues,
            "low_quality": bool(issues),
            "quality_score": worst,
            # Raw scores, not probabilities: these issues get no detection_confidence tag.
            "scores": {"topiq": worst, "topiq_mean": round(r[TOPIQ], 4)},
            "view_scores": r.get(f"{TOPIQ}_views"),  # one per view; four for a 360° picture
            "model": self.model_id,
        }
