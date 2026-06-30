from industrial_world_model.data.manifest import build_manifest, scan_dataset


def test_manifest_entry_for_missing_dataset(tmp_path):
    entry = scan_dataset("missing", tmp_path / "missing", "visual")
    assert entry.dataset == "missing"
    assert entry.exists is False
    assert entry.usable_for_visual_anomaly is False


def test_default_manifest_contains_core_datasets():
    entries = build_manifest()
    names = {e.dataset for e in entries}
    assert "mvtec_ad" in names
    assert "cnc_milling" in names
