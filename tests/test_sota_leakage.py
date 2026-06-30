from common.config import load_config
from sensor_jepa.data.cnc_world_model import prepare_transition_from_config
from sensor_jepa.train.sota_benchmark import leakage_report


def test_leakage_report_tool_split_passes():
    cfg = load_config("configs/sensor_jepa/demo_sensor_quick.yaml")
    bundle = prepare_transition_from_config(cfg)
    report = leakage_report(bundle)
    assert report["passes"] is True
    assert not report["overlap"]["train_test"]

