import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


def _load_benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "41_visual_foundation_benchmark.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("visual_foundation_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_two_categories_are_partitioned_before_any_fit_or_split():
    module = _load_benchmark_module()
    records = [
        SimpleNamespace(category="bottle", sample_id="b1"),
        SimpleNamespace(category="cable", sample_id="c1"),
        SimpleNamespace(category="bottle", sample_id="b2"),
        SimpleNamespace(category="cable", sample_id="c2"),
    ]
    partitions = module._partition_records_by_category(records)
    assert list(partitions) == ["bottle", "cable"]
    assert {record.category for record in partitions["bottle"]} == {"bottle"}
    assert {record.category for record in partitions["cable"]} == {"cable"}
    assert not (set(map(id, partitions["bottle"])) & set(map(id, partitions["cable"])))


def test_multicategory_dispatch_calls_split_once_per_isolated_category(monkeypatch, tmp_path):
    module = _load_benchmark_module()
    all_records = [SimpleNamespace(category="bottle"), SimpleNamespace(category="cable")]
    split_scopes = []

    def fake_collect(_root, *, dataset, categories, max_samples):
        if categories is None or len(categories) > 1:
            return all_records
        return [record for record in all_records if record.category == categories[0]]

    def fake_split(records):
        split_scopes.append({record.category for record in records})
        return [], []  # stop each leaf before any image/model work

    monkeypatch.setattr(module, "collect_mvtec_records", fake_collect)
    monkeypatch.setattr(module, "split_records", fake_split)
    result = module.run(
        {"dataset": {"categories": "bottle,cable"}}, tmp_path,
    )
    assert split_scopes == [{"bottle"}, {"cable"}]
    assert set(result["category"]) == {"bottle", "cable"}
