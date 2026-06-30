from __future__ import annotations

import argparse
from pathlib import Path

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.data.manifest import build_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Industrial World Model dataset manifest.")
    parser.add_argument("--out-root", default="outputs/industrial_world_model")
    args = parser.parse_args()
    out = Path(args.out_root)
    entries = build_manifest()
    save_manifest(entries, out / "dataset_manifest.json", out / "dataset_manifest.md")
    print(f"Manifest written to {out / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
