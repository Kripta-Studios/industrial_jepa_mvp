from common.config import load_config
from sensor_jepa.data.cnc_world_model import prepare_transition_from_config


def test_cnc_transition_bundle_shapes():
    cfg = load_config("configs/sensor_jepa/demo_sensor_quick.yaml")
    bundle = prepare_transition_from_config(cfg)
    assert bundle.x_train.shape[0] == bundle.a_train.shape[0] == bundle.x_next_train.shape[0]
    assert bundle.action_dim >= 3
    assert set(bundle.y_failure_test.tolist()).issubset({0, 1})

