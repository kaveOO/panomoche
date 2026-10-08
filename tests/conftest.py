import numpy as np
import pytest
from PIL import Image


def textured(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Street-ish test picture: smooth gradients + sharp edges + fine texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([xx / w, yy / h, 0.5 + 0.5 * np.sin(xx / 37.0)], -1) * 160 + 40
    checker = ((xx // 24 + yy // 24) % 2)[..., None] * 50
    noise = rng.normal(0, 12, (h, w, 3))
    return np.clip(base + checker + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def flat_rgb():
    return textured(640, 480)


@pytest.fixture
def make_jpegs(tmp_path):
    def _make(n: int, size=(800, 600)) -> list:
        paths = []
        for i in range(n):
            p = tmp_path / f"pic{i:03d}.jpg"
            Image.fromarray(textured(*size, seed=i)).save(p, quality=92)
            paths.append(p)
        return paths

    return _make


@pytest.fixture(scope="session")
def tiny_backbone():
    """Randomly initialised DINOv3 ViT-S: exercises the real code path without downloading weights."""
    from panomoche.backbone import DinoBackbone

    return DinoBackbone(pretrained=False, threads=2)
