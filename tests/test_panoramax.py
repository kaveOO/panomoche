from panomoche.panoramax import item_instance_api, item_is_pano, item_record, semantics_for

ITEM = {
    "id": "9a479835-1331-4b8a-adce-f2173f61ffc8",
    "collection": "600d5cee-5328-4dd4-8838-5f27ef85aa59",
    "geometry": {"type": "Point", "coordinates": [2.345, 48.859]},
    "links": [
        {"rel": "root", "href": "https://api.panoramax.xyz/api"},
        {"rel": "via", "href": "https://panoramax.ign.fr", "instance_name": "ign"},
    ],
    "properties": {
        "license": "etalab-2.0",
        "geovisio:rank_in_collection": 5907,
        "pers:interior_orientation": {
            "camera_manufacturer": "IMAJING",
            "camera_model": "imajbox 360 HD",
            "field_of_view": 360,
        },
    },
}


def test_writes_target_the_hosting_instance_not_the_meta_catalog():
    assert item_instance_api(ITEM) == "https://panoramax.ign.fr/api"
    local = {**ITEM, "links": [{"rel": "root", "href": "https://panoramax.openstreetmap.fr/api"}]}
    assert item_instance_api(local) == "https://panoramax.openstreetmap.fr/api"


def test_pano_detection_from_field_of_view():
    assert item_is_pano(ITEM) is True
    flat = {"properties": {"pers:interior_orientation": {"field_of_view": 96}}}
    assert item_is_pano(flat) is False
    assert item_is_pano({"properties": {}}) is None


def test_item_record():
    r = item_record(ITEM)
    assert r["camera"] == "IMAJING imajbox 360 HD" and r["is_pano"] and r["instance_api"].endswith("ign.fr/api")
    assert r["rank"] == 5907


def test_semantics_follow_panoramax_qualifier_convention():
    pred = {"issues": ["sequence_outlier"], "scores": {"topiq": 0.3}, "model": "panomoche-topiq_nr-spaq/0.1.0"}
    assert semantics_for(pred) == [
        {"key": "quality_issue", "value": "sequence_outlier", "action": "add"},
        {
            "key": "detection_model[quality_issue=sequence_outlier]",
            "value": "panomoche-topiq_nr-spaq/0.1.0",
            "action": "add",
        },
    ]
    assert semantics_for({**pred, "issues": []}) == []
    assert semantics_for(pred, tag_key="q")[1]["key"] == "detection_model[q=sequence_outlier]"


def test_random_sample_takes_one_picture_per_sequence(monkeypatch):
    from panomoche.panoramax import PanoramaxClient

    def fake_search(self, bbox=None, limit=100, **kw):
        if bbox[0] < 0:  # western half of the region: no coverage
            return []
        cols = ["a", "b"] if bbox[1] < 46 else ["a", "c"]
        return [{"id": f"{c}-{i}-{bbox[0]:.3f}", "collection": c} for c in cols for i in range(3)]

    monkeypatch.setattr(PanoramaxClient, "search", fake_search)
    items = PanoramaxClient(delay=0).random_sample(5, seed=1)
    assert len(items) == 3 and len({i["collection"] for i in items}) == 3  # only 3 sequences exist
