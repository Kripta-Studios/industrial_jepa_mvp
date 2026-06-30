from __future__ import annotations

import _bootstrap  # noqa: F401
from common.manifest import write_dataset_manifest


def main() -> None:
    write_dataset_manifest()
    print("No downloads are performed by this MVP. Local dataset manifest refreshed.")


if __name__ == "__main__":
    main()

