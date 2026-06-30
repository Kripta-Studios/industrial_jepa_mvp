from sensor_jepa.models.official_time_series_baselines import baseline_metadata_for_name, check_official_baseline_availability


def test_baseline_availability_reports_fallbacks_as_not_sota_eligible():
    availability = check_official_baseline_availability()
    assert isinstance(availability.fallback_enabled, bool)
    meta = baseline_metadata_for_name("minirocket_lite", "rocket_fallback", "fallback")
    assert meta["fallback_used"] is True
    assert meta["sota_claim_eligible"] is False
