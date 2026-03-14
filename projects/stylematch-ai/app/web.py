from __future__ import annotations

from functools import lru_cache
import os
import socket
from typing import Any

from flask import Flask, render_template, request

from app.catalog import load_catalog, row_to_text
from app.config import settings
from app.embedding import EmbeddingService
from app.endee_client import EndeeClient
from app.recommend import build_filters, safe_json_load


app = Flask(__name__)


@lru_cache(maxsize=1)
def get_catalog():
    return load_catalog()


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingService:
    return EmbeddingService(settings.embedding_model)


def parse_text_filters(style: str | None, occasion: str | None, clothing_types: str | None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if occasion:
        filters.append({"occasion": {"$eq": occasion}})
    if style:
        filters.append({"style": {"$eq": style}})
    if clothing_types:
        values = [value.strip() for value in clothing_types.split(",") if value.strip()]
        if values:
            filters.append({"clothing_type": {"$in": values}})
    return filters


@app.route("/", methods=["GET", "POST"])
def home():
    results: list[dict[str, Any]] = []
    error = ""
    info_message = ""
    selected_item_summary = ""
    catalog = get_catalog()
    item_options = [
        {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "type": str(row["clothing_type"]),
            "style": str(row["style"]),
        }
        for _, row in catalog.iterrows()
    ]

    form_data = {
        "mode": "text",
        "query": "",
        "item_id": "",
        "item_pick": "",
        "style": "",
        "occasion": "",
        "clothing_types": "",
        "k": str(settings.top_k),
    }

    if request.method == "POST":
        mode = request.form.get("mode", "text").strip().lower()
        query = request.form.get("query", "").strip()
        item_id = request.form.get("item_id", "").strip()
        item_pick = request.form.get("item_pick", "").strip()
        style = request.form.get("style", "").strip() or None
        occasion = request.form.get("occasion", "").strip() or None
        clothing_types = request.form.get("clothing_types", "").strip() or None

        if item_pick and mode == "text" and not query:
            mode = "item"

        if not item_id and item_pick:
            item_id = item_pick

        try:
            k = int(request.form.get("k", str(settings.top_k)))
            k = max(1, min(k, 20))
        except ValueError:
            k = settings.top_k

        form_data.update(
            {
                "mode": mode,
                "query": query,
                "item_id": item_id,
                "item_pick": item_pick,
                "style": style or "",
                "occasion": occasion or "",
                "clothing_types": clothing_types or "",
                "k": str(k),
            }
        )

        client = EndeeClient(settings.endee_base_url, settings.endee_auth_token)
        embedder = get_embedder()

        try:
            if mode == "item":
                if not item_id:
                    raise ValueError("Enter an item ID for item-based recommendations.")

                selected = catalog[catalog["id"] == str(item_id)]
                if selected.empty:
                    raise ValueError(f"Item id {item_id} not found in catalog.csv")

                row = selected.iloc[0]
                selected_item_summary = f"Input item: {row['name']} ({row['clothing_type']}, {row['style']})"
                query_text = row_to_text(row)
                if style or occasion:
                    preference_bits = []
                    if style:
                        preference_bits.append(f"preferred style {style}")
                    if occasion:
                        preference_bits.append(f"for {occasion}")
                    query_text = f"{query_text} User preference: {' '.join(preference_bits)}."

                query_vector = embedder.encode_one(query_text)
                strict_filters = build_filters(row.to_dict(), style, occasion)

                search_results = client.search(
                    index_name=settings.index_name,
                    query_vector=query_vector,
                    k=k + 3,
                    filters=strict_filters,
                )

                if not search_results and (style or occasion):
                    relaxed_filters = build_filters(row.to_dict(), None, None)
                    search_results = client.search(
                        index_name=settings.index_name,
                        query_vector=query_vector,
                        k=k + 3,
                        filters=relaxed_filters,
                    )
                    if search_results:
                        info_message = "No exact match for selected style/occasion. Showing closest compatible outfit matches."

                if not search_results:
                    search_results = client.search(
                        index_name=settings.index_name,
                        query_vector=query_vector,
                        k=k + 3,
                        filters=None,
                    )
                    if search_results:
                        info_message = "No strict compatible match found. Showing nearest items from catalog."

                if not search_results:
                    info_message = "No recommendations found for this item. Try a different item or remove filters."

                for result in search_results:
                    if str(result.get("id")) == str(item_id):
                        continue
                    meta = safe_json_load(result.get("meta"))
                    results.append(
                        {
                            "id": result.get("id", ""),
                            "score": float(result.get("similarity", 0.0)),
                            "name": meta.get("name", "N/A"),
                            "type": meta.get("clothing_type", "N/A"),
                            "style": meta.get("style", "N/A"),
                            "color": meta.get("color", "N/A"),
                            "occasion": meta.get("occasion", "N/A"),
                        }
                    )
                    if len(results) >= k:
                        break

            else:
                if not query:
                    raise ValueError("Enter a query in text mode.")

                query_vector = embedder.encode_one(query)
                filters = parse_text_filters(style, occasion, clothing_types)
                search_results = client.search(
                    index_name=settings.index_name,
                    query_vector=query_vector,
                    k=k,
                    filters=filters,
                )

                if not search_results and filters:
                    search_results = client.search(
                        index_name=settings.index_name,
                        query_vector=query_vector,
                        k=k,
                        filters=None,
                    )
                    if search_results:
                        info_message = "No exact match for selected filters. Showing closest matches instead."

                if not search_results:
                    info_message = "No recommendations found. Try fewer filters or different query text."

                for result in search_results:
                    meta = safe_json_load(result.get("meta"))
                    results.append(
                        {
                            "id": result.get("id", ""),
                            "score": float(result.get("similarity", 0.0)),
                            "name": meta.get("name", "N/A"),
                            "type": meta.get("clothing_type", "N/A"),
                            "style": meta.get("style", "N/A"),
                            "color": meta.get("color", "N/A"),
                            "occasion": meta.get("occasion", "N/A"),
                        }
                    )

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        results=results,
        error=error,
        info_message=info_message,
        selected_item_summary=selected_item_summary,
        form=form_data,
        index_name=settings.index_name,
        item_options=item_options,
    )


def main() -> None:
    start_port = int(os.getenv("STYLEMATCH_UI_PORT", os.getenv("PORT", "5000")))
    debug_mode = os.getenv("STYLEMATCH_UI_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    chosen_port = start_port

    for candidate in range(start_port, start_port + 25):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", candidate)) != 0:
                chosen_port = candidate
                break

    if chosen_port != start_port:
        print(f"Port {start_port} is busy. Starting UI on port {chosen_port} instead.")

    app.run(host="0.0.0.0", port=chosen_port, debug=debug_mode, use_reloader=debug_mode)


if __name__ == "__main__":
    main()
