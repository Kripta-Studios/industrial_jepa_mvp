import subprocess
import sys
import json
from pathlib import Path


def test_product_demo_generation(tmp_path):
    out = tmp_path / "demo"
    result = subprocess.run(
        [sys.executable, "scripts/56_build_product_demo.py", "--out-root", str(out), "--results-root", str(tmp_path / "missing")],
        check=True,
        capture_output=True,
        text=True,
    )
    html = out / "index.html"
    data = out / "demo_data.json"
    assert html.exists()
    assert data.exists()
    text = html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert "Industrial Predictive Quality World Model" in text
    payload = json.loads(data.read_text(encoding="utf-8"))
    assert payload["mode"] == "smoke"
    assert payload["claim_tier"] == "smoke"
    assert payload["comparable_benchmark"] is False
    assert result.returncode == 0
