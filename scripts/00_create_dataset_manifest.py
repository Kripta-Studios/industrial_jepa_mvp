from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.manifest import write_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-yaml", default="data/manifests/datasets.yaml")
    parser.add_argument("--out-csv", default="data/manifests/datasets.csv")
    args = parser.parse_args()
    yml, csv = write_dataset_manifest(args.raw_root, args.out_yaml, args.out_csv)
    print(f"Wrote {yml}")
    print(f"Wrote {csv}")


if __name__ == "__main__":
    main()

