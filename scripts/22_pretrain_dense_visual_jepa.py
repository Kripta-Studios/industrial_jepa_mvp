from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from visual_jepa.train.pretrain_dense_jepa import pretrain_dense_visual_jepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/dense_mvtec_bottle_quick.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ckpt, history = pretrain_dense_visual_jepa(cfg)
    print(f"Saved {ckpt}; final_loss={history[-1]['loss']:.6f}; val_loss={history[-1]['val_loss']:.6f}")


if __name__ == "__main__":
    main()
