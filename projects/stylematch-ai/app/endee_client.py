from __future__ import annotations

import json
from typing import Any

import msgpack
import requests


class EndeeClient:
    def __init__(self, base_url: str, auth_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if auth_token:
            self.session.headers.update({"Authorization": auth_token})

    def create_index(self, index_name: str, dim: int, space_type: str = "cosine") -> None:
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "precision": "float32",
            "M": 32,
            "ef_con": 200,
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/index/create",
            json=payload,
            timeout=30,
        )

        if response.status_code in (200, 409):
            return
        response.raise_for_status()

    def list_indexes(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/api/v1/index/list", timeout=15)
        response.raise_for_status()
        return response.json()

    def insert_vectors(self, index_name: str, vectors: list[dict[str, Any]]) -> None:
        response = self.session.post(
            f"{self.base_url}/api/v1/index/{index_name}/vector/insert",
            json=vectors,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()

    def search(
        self,
        index_name: str,
        query_vector: list[float],
        k: int,
        filters: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "vector": query_vector,
            "k": k,
            "include_vectors": False,
        }
        if filters:
            payload["filter"] = json.dumps(filters)

        response = self.session.post(
            f"{self.base_url}/api/v1/index/{index_name}/search",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        unpacked = msgpack.unpackb(response.content, raw=False)
        if isinstance(unpacked, dict):
            return unpacked.get("results", [])

        # Endee may return ResultSet as MessagePack array of positional fields:
        # [similarity, id, meta, filter, norm, vector]
        if isinstance(unpacked, list):
            parsed: list[dict[str, Any]] = []
            for row in unpacked:
                if isinstance(row, dict):
                    parsed.append(row)
                    continue

                if isinstance(row, (list, tuple)) and len(row) >= 6:
                    parsed.append(
                        {
                            "similarity": row[0],
                            "id": row[1],
                            "meta": row[2],
                            "filter": row[3],
                            "norm": row[4],
                            "vector": row[5],
                        }
                    )
            return parsed

        return []
