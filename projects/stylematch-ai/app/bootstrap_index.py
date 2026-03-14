from __future__ import annotations

from app.catalog import load_catalog, row_to_filter_json, row_to_meta_json, row_to_text
from app.config import settings
from app.embedding import EmbeddingService
from app.endee_client import EndeeClient


def main() -> None:
    catalog = load_catalog()
    embedder = EmbeddingService(settings.embedding_model)
    client = EndeeClient(settings.endee_base_url, settings.endee_auth_token)

    vectors = embedder.encode(catalog.apply(row_to_text, axis=1).tolist())
    client.create_index(index_name=settings.index_name, dim=embedder.dim())

    payload = []
    for (_, row), vector in zip(catalog.iterrows(), vectors):
        payload.append(
            {
                "id": str(row["id"]),
                "meta": row_to_meta_json(row),
                "filter": row_to_filter_json(row),
                "vector": vector,
            }
        )

    client.insert_vectors(settings.index_name, payload)
    print(f"Indexed {len(payload)} products into '{settings.index_name}'.")


if __name__ == "__main__":
    main()
