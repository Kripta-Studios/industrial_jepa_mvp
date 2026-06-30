from pathlib import Path

from common.manifest import write_dataset_manifest


def test_dataset_manifest_writes(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "sensor" / "cnc_milling").mkdir(parents=True)
    (raw / "sensor" / "cnc_milling" / "x.csv").write_text("a\n1\n", encoding="utf-8")
    yml, csv = write_dataset_manifest(raw, tmp_path / "datasets.yaml", tmp_path / "datasets.csv")
    assert yml.exists()
    assert csv.exists()

