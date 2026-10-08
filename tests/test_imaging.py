from PIL import Image

from panomoche.imaging import PANO_VIEWS, VIEW_SIZE, prepare

from .conftest import textured


def test_flat_picture_is_downscaled_never_upscaled():
    big = prepare(Image.fromarray(textured(2048, 1536)))
    assert not big.is_pano and big.rgb.shape == (384, 512, 3) and len(big.views()) == 1
    small = prepare(Image.fromarray(textured(300, 200)))
    assert small.rgb.shape == (200, 300, 3)


def test_equirectangular_keeps_horizon_band_split_in_views():
    p = prepare(Image.fromarray(textured(4096, 2048)))
    assert p.is_pano
    assert p.rgb.shape == (VIEW_SIZE, PANO_VIEWS * VIEW_SIZE, 3)
    assert [v.shape for v in p.views()] == [(VIEW_SIZE, VIEW_SIZE, 3)] * PANO_VIEWS


def test_pano_hint_overrides_aspect_ratio():
    assert not prepare(Image.fromarray(textured(1024, 512)), is_pano=False).is_pano
