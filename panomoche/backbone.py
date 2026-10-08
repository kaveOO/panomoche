"""DINOv3 feature extractor.

Weights come from timm, which redistributes the official DINOv3 LVD-1689M
checkpoints without the manual-approval gate of ``facebook/dinov3-*`` on the
Hugging Face hub. The DINOv3 License still applies to them.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

DEFAULT_BACKBONE = "vit_small_patch16_dinov3.lvd1689m"


class DinoBackbone:
    def __init__(
        self,
        name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        device: str | None = None,
        threads: int | None = None,
    ):
        import timm

        torch.set_num_threads(threads or max(1, (os.cpu_count() or 2) // 2))
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True).eval().to(self.device)
        )
        cfg = timm.data.resolve_model_data_config(self.model)
        self.mean = np.array(cfg["mean"], dtype=np.float32) * 255
        self.std = np.array(cfg["std"], dtype=np.float32) * 255
        self.patch = self.model.patch_embed.patch_size[0]
        self.n_prefix = self.model.num_prefix_tokens  # CLS + register tokens

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        h, w = rgb.shape[:2]
        th, tw = (
            max(self.patch, round(h / self.patch) * self.patch),
            max(self.patch, round(w / self.patch) * self.patch),
        )
        if (th, tw) != (h, w):
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1))

    @torch.inference_mode()
    def embed_views(self, views: list[np.ndarray]) -> np.ndarray:
        """Per-view [CLS, mean patch token], each L2-normalised. Shape (n_views, 2*dim)."""
        tensors = [self._to_tensor(v) for v in views]
        out: list[np.ndarray | None] = [None] * len(views)
        by_shape: dict[tuple, list[int]] = {}
        for i, t in enumerate(tensors):
            by_shape.setdefault(tuple(t.shape), []).append(i)
        for idx in by_shape.values():
            tokens = self.model.forward_features(torch.stack([tensors[i] for i in idx]).to(self.device))
            cls = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
            patches = torch.nn.functional.normalize(tokens[:, self.n_prefix :].mean(1), dim=-1)
            feats = torch.cat([cls, patches], dim=-1).float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = feats[j]
        return np.stack(out)
