from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import ensure_dir


def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)
    return path


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "\n_No rows._\n"
    df = pd.DataFrame(rows)
    cols = [str(c) for c in df.columns]

    def fmt(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    matrix = [[fmt(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(cols[i]), *(len(row[i]) for row in matrix)) if matrix else len(cols[i])
        for i in range(len(cols))
    ]
    header = "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in matrix]
    return "\n".join([header, sep, *body])


def write_markdown_report(
    path: str | Path,
    title: str,
    sections: dict[str, str],
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
