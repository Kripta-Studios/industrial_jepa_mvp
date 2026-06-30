from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    run([sys.executable, "scripts/15_run_sensor_demo.py", "--config", "configs/sensor_jepa/demo_sensor_quick.yaml"])
    run([sys.executable, "scripts/16_run_visual_demo.py", "--config", "configs/visual_jepa/demo_visual_quick.yaml"])
    run([sys.executable, "scripts/12_compare_experiments.py"])


if __name__ == "__main__":
    main()

