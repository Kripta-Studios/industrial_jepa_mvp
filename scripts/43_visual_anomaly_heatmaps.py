from __future__ import annotations

import argparse
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual anomaly heatmaps via benchmark script.")
    parser.add_argument("--config", default="configs/industrial_world_model/visual_foundation_mvtec.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/visual_foundation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--categories", default="bottle")
    parser.parse_args()
    import runpy

    runpy.run_path("scripts/41_visual_foundation_benchmark.py", run_name="__main__")


if __name__ == "__main__":
    main()
