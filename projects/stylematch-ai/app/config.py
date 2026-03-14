from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    endee_base_url: str = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
    endee_auth_token: str = os.getenv("ENDEE_AUTH_TOKEN", "")
    index_name: str = os.getenv("INDEX_NAME", "stylematch_outfits")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "6"))


settings = Settings()
