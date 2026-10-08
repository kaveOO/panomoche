import numpy as np
from PIL import Image

from panomoche.imaging import prepare
from panomoche.orientation import ROTATIONS, is_wrong, predict, rotate, train

from .conftest import textured


def test_rotate_round_trip():
    v = textured(64, 48)
    assert rotate(v, 90).shape == (64, 48, 3)
    assert np.array_equal(rotate(rotate(v, 180), 180), v)


def test_decision_rule():
    assert is_wrong({"orientation": 180, "orientation_confidence": 0.995})
    assert not is_wrong({"orientation": 180, "orientation_confidence": 0.98})
    assert not is_wrong({"orientation": 0, "orientation_confidence": 0.99})
    assert not is_wrong(None)


def test_train_and_predict_plumbing(tiny_backbone):
    preps = [prepare(Image.fromarray(textured(320, 240, seed=i))) for i in range(4)]
    clf = train(tiny_backbone, preps)
    r = predict(tiny_backbone, clf, prepare(Image.fromarray(textured(2048, 1024))))  # 360°: 4 views averaged
    assert r["orientation"] in ROTATIONS and 0 <= r["orientation_confidence"] <= 1


def test_service_excludes_upside_down_pictures():
    import io

    from fastapi.testclient import TestClient

    from panomoche.service import create_app

    from .test_service import INFO, fake_score

    class AlwaysUpsideDown:
        def predict_proba(self, X):
            return np.tile([0.002, 0.001, 0.996, 0.001], (len(X), 1))

    class NoBackbone:
        def embed_views(self, views):
            return np.zeros((len(views), 4))

    client = TestClient(create_app(fake_score, INFO, orientation=(NoBackbone(), AlwaysUpsideDown())))
    buf = io.BytesIO()
    Image.fromarray(textured(640, 480)).save(buf, format="JPEG")
    body = client.post("/predict", files={"picture": ("a.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert body["excluded"] == "wrong_orientation" and body["orientation"] == 180


def test_perspective_view_of_an_equirectangular_picture():
    from panomoche.orientation import perspective

    eq = np.zeros((512, 1024, 3), np.uint8)
    eq[:256] = 255  # sky (top half) white, ground black
    eq[:, 512:] = eq[:, 512:] // 2 + 60  # the back half is tinted, to check yaw
    ahead, behind = perspective(eq, 0, size=64), perspective(eq, 180, size=64)
    assert ahead[:20].mean() > 200 and ahead[-20:].mean() < 50  # level camera: sky up, ground down
    assert not np.array_equal(ahead, behind)


def test_360_rule_needs_two_rotated_views():
    two = {
        "orientation": 90,
        "orientation_confidence": 0.995,
        "orientation_views": [[270, 0.999], [90, 0.995], [0, 0.9], [180, 0.6]],
    }
    one = {
        "orientation": 0,
        "orientation_confidence": 0.6,
        "orientation_views": [[0, 0.999], [90, 0.997], [0, 0.99], [0, 0.98]],
    }  # a close wall seen sideways
    weak = {
        "orientation": 270,
        "orientation_confidence": 0.98,
        "orientation_views": [[270, 0.98], [270, 1.0], [180, 0.97], [90, 0.98]],
    }
    assert is_wrong(two) and not is_wrong(one) and not is_wrong(weak)


def test_360_pictures_are_judged_on_rendered_views(tiny_backbone):
    preps = [prepare(Image.fromarray(textured(320, 240, seed=i))) for i in range(4)]
    clf = train(tiny_backbone, preps)
    img = Image.fromarray(textured(2048, 1024))
    r = predict(tiny_backbone, clf, prepare(img), image=img)
    assert len(r["orientation_views"]) == 4 and r["orientation"] in ROTATIONS


def test_service_excludes_pitched_360_picture():
    import io

    from fastapi.testclient import TestClient

    from panomoche.service import create_app

    from .test_service import INFO, fake_score

    class SideViewsSideways:  # views ahead / right / behind / left
        def predict_proba(self, X):
            return np.array(
                [
                    [0.99, 0.004, 0.003, 0.003],
                    [0.001, 0.997, 0.001, 0.001],
                    [0.99, 0.004, 0.003, 0.003],
                    [0.001, 0.001, 0.001, 0.997],
                ]
            )[: len(X)]

    class NoBackbone:
        def embed_views(self, views):
            return np.zeros((len(views), 4))

    client = TestClient(create_app(fake_score, INFO, orientation=(NoBackbone(), SideViewsSideways())))
    buf = io.BytesIO()
    Image.fromarray(textured(2048, 1024)).save(buf, format="JPEG")
    body = client.post("/predict", files={"picture": ("p.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert body["is_pano"] and body["excluded"] == "wrong_orientation"
    assert [v[0] for v in body["orientation_views"]] == [0, 90, 0, 270]
