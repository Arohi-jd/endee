from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT_DIR / "data" / "catalog.csv"


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    return df


def row_to_text(row: pd.Series) -> str:
    # Text representation used for semantic embedding.
    return (
        f"{row['name']}. "
        f"Type: {row['clothing_type']}. "
        f"Style: {row['style']}. "
        f"Color: {row['color']}. "
        f"Occasion: {row['occasion']}. "
        f"Gender: {row['gender']}. "
        f"Season: {row['season']}."
    )


def row_to_filter_json(row: pd.Series) -> str:
    payload: dict[str, Any] = {
        "clothing_type": row["clothing_type"],
        "style": row["style"],
        "color": row["color"],
        "occasion": row["occasion"],
        "gender": row["gender"],
        "season": row["season"],
        "price": int(row["price"]),
    }
    return json.dumps(payload)


def row_to_meta_json(row: pd.Series) -> str:
    payload: dict[str, Any] = {
        "name": row["name"],
        "price": float(row["price"]),
        "clothing_type": row["clothing_type"],
        "style": row["style"],
        "color": row["color"],
        "occasion": row["occasion"],
        "gender": row["gender"],
        "season": row["season"],
    }
    return json.dumps(payload)
