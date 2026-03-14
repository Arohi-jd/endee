from __future__ import annotations

import argparse
import json
from typing import Any

from app.catalog import load_catalog, row_to_text
from app.config import settings
from app.embedding import EmbeddingService
from app.endee_client import EndeeClient


COMPATIBILITY = {
    "top": ["bottom", "outerwear", "shoes", "accessory"],
    "bottom": ["top", "outerwear", "shoes", "accessory"],
    "outerwear": ["top", "bottom", "shoes", "accessory"],
    "onepiece": ["outerwear", "shoes", "accessory"],
    "shoes": ["top", "bottom", "onepiece", "outerwear"],
    "accessory": ["top", "bottom", "onepiece", "outerwear"],
}


def safe_json_load(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def build_filters(item: dict[str, Any], style: str | None, occasion: str | None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []

    compatible_types = COMPATIBILITY.get(item["clothing_type"], ["top", "bottom", "shoes"])
    filters.append({"clothing_type": {"$in": compatible_types}})

    target_occasion = occasion or item["occasion"]
    filters.append({"occasion": {"$eq": target_occasion}})

    if style:
        filters.append({"style": {"$eq": style}})

    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend matching outfits with Endee vector search")
    parser.add_argument("--item-id", required=True, help="Catalog item ID to build outfit around")
    parser.add_argument("--style", help="Optional preferred style override")
    parser.add_argument("--occasion", help="Optional occasion override")
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()

    catalog = load_catalog()
    selected = catalog[catalog["id"] == str(args.item_id)]
    if selected.empty:
        raise ValueError(f"Item id {args.item_id} not found in catalog.csv")

    row = selected.iloc[0]
    item = row.to_dict()

    preference_text = []
    if args.style:
        preference_text.append(f"preferred style {args.style}")
    if args.occasion:
        preference_text.append(f"for {args.occasion}")

    query_text = row_to_text(row)
    if preference_text:
        query_text = f"{query_text} User preference: {' '.join(preference_text)}."

    embedder = EmbeddingService(settings.embedding_model)
    query_vector = embedder.encode_one(query_text)

    filters = build_filters(item, args.style, args.occasion)

    client = EndeeClient(settings.endee_base_url, settings.endee_auth_token)
    results = client.search(
        index_name=settings.index_name,
        query_vector=query_vector,
        k=args.k + 3,
        filters=filters,
    )

    print(f"\nInput item: {row['name']} ({row['clothing_type']}, {row['style']})")
    print("Recommended matches:\n")

    shown = 0
    for result in results:
        if result.get("id") == str(args.item_id):
            continue

        meta = safe_json_load(result.get("meta"))
        print(
            f"- id={result.get('id')} | score={result.get('similarity', 0):.4f} | "
            f"name={meta.get('name', 'N/A')} | type={meta.get('clothing_type', 'N/A')} | "
            f"style={meta.get('style', 'N/A')} | color={meta.get('color', 'N/A')}"
        )
        shown += 1
        if shown >= args.k:
            break


if __name__ == "__main__":
    main()
