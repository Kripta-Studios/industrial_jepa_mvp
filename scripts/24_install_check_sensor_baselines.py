from __future__ import annotations

import _bootstrap  # noqa: F401
from sensor_jepa.models.official_time_series_baselines import diagnostic_lines


def main() -> None:
    for line in diagnostic_lines():
        print(line)


if __name__ == "__main__":
    main()
