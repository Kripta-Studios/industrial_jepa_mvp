from sensor_jepa.models.sensor_jepa import SensorJEPA


def test_freeze_encoder_parameters():
    model = SensorJEPA(input_channels=4, embedding_dim=8, hidden_dim=8)
    for p in model.encoder.parameters():
        p.requires_grad = False
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.predictor.parameters())

