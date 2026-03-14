# StyleMatch AI — Outfit Recommendation with Endee Vector Search

StyleMatch AI is an AI/ML recommendation project that suggests matching clothing combinations using embedding similarity and metadata-aware retrieval.

It uses **Endee** as the vector database for:

- storing product embeddings,
- fast nearest-neighbor search,
- filter-aware retrieval (style, color, occasion, clothing type, user preference constraints).

---

## 1) Problem Statement

Traditional outfit recommendation engines rely on static rules (for example: “white shirt + blue jeans”).

StyleMatch AI improves this by using semantic embeddings and vector search to capture nuanced compatibility signals across:

- clothing type,
- style,
- color,
- occasion,
- user preference.

The result is a recommendation flow that is both flexible and explainable.

---

## 2) Practical Use Case

This is a **recommendation system** where vector search is the core retrieval primitive.

Given one item (for example a blazer), the system:

1. Builds a semantic query embedding from item attributes + optional preference text.
2. Retrieves similar/compatible vectors from Endee.
3. Applies metadata filters (occasion, style, compatible clothing categories).
4. Returns top-k matching items for outfit composition.

---

## 3) System Design and Technical Approach

### 3.1 Architecture

1. **Catalog Loader**
   - Reads the fashion dataset from [data/catalog.csv](data/catalog.csv).
2. **Embedding Service**
   - Uses Sentence Transformers (`all-MiniLM-L6-v2`) to encode each product description.
3. **Endee Indexing Pipeline**
   - Creates a vector index in Endee.
   - Inserts vectors with `meta` and `filter` payloads.
4. **Recommendation API/CLI Layer**
   - Builds query embedding.
   - Calls Endee similarity search.
   - Decodes MessagePack results.
   - Prints ranked recommendations.

### 3.2 Retrieval Logic

- Dense vector similarity captures semantic closeness.
- Metadata filters enforce practical constraints:
  - `clothing_type` compatibility (`top` -> `bottom`, `outerwear`, etc.),
  - `occasion` equality,
  - optional style overrides.

### 3.3 Why this is AI/ML

- Uses a pretrained transformer embedding model.
- Converts structured fashion attributes into semantic vector space.
- Uses ANN/vector similarity retrieval as the ranking backbone.

---

## 4) How Endee is Used

Endee is used as the primary retrieval engine through these APIs:

- `POST /api/v1/index/create` → create the `stylematch_outfits` index.
- `POST /api/v1/index/{index_name}/vector/insert` → store embedding vectors + metadata.
- `POST /api/v1/index/{index_name}/search` → retrieve top-k nearest vectors with filters.

In this project:

- `filter` payload is stored with each vector (JSON string), e.g.:
  - `{"style":"formal","occasion":"office","color":"navy"}`
- Query-time filter uses Endee filter array syntax, e.g.:
  - `[{"occasion":{"$eq":"office"}}, {"clothing_type":{"$in":["bottom","shoes"]}}]`

---

## 5) Mandatory Repository Usage Steps (Evaluation Compliance)

Before starting your own submission, complete these required steps:

1. Star the official Endee repository:
   - [https://github.com/endee-io/endee](https://github.com/endee-io/endee)
2. Fork it to your personal GitHub account.
3. Build your project on top of your forked repository.

> These steps are mandatory per evaluation rules.

---

## 6) Setup and Execution

### 6.1 Prerequisites

- Python 3.10+
- Endee server running on `http://localhost:8080`
  - See root docs: [docs/getting-started.md](../../docs/getting-started.md)

### 6.2 Project Setup

From the project directory:

1. Create environment and install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`

2. Configure env:
   - `cp .env.example .env`
   - set `ENDEE_BASE_URL`, `ENDEE_AUTH_TOKEN` (if enabled)

### 6.3 Build Index + Insert Data

- `python -m app.bootstrap_index`

This will:

- load [data/catalog.csv](data/catalog.csv),
- generate sentence embeddings,
- create Endee index,
- insert vectors with metadata/filter payload.

### 6.4 Run Recommendation Query

- From the repo root, move into the project folder:
   - `cd projects`
   - `cd stylematch-ai`

- Run recommendation query:
   - `python -m app.recommend --item-id 2 --k 5`

- Expected outcome:

```text
Input item: Navy Tailored Blazer (outerwear, formal)
Recommended matches:

- id=22 | score=0.3142 | name=Navy Pleated Trousers | type=bottom | style=formal | color=navy
- id=29 | score=0.4458 | name=Tan Wide-Leg Pants | type=bottom | style=minimal | color=tan
- id=4 | score=0.4772 | name=Black Leather Derby Shoes | type=shoes | style=formal | color=black
- id=1 | score=0.4979 | name=Classic White Oxford Shirt | type=top | style=smart_casual | color=white
- id=21 | score=0.5000 | name=Ivory Silk Blouse | type=top | style=formal | color=ivory
```

- Optional preference override:
  - `python -m app.recommend --item-id 2 --style formal --occasion office --k 5`

---

## 7) Repository Structure

- [app/bootstrap_index.py](app/bootstrap_index.py): index creation + ingestion pipeline
- [app/recommend.py](app/recommend.py): query and recommendation logic
- [app/embedding.py](app/embedding.py): transformer embedding wrapper
- [app/endee_client.py](app/endee_client.py): Endee API client
- [app/catalog.py](app/catalog.py): catalog transformation utilities
- [data/catalog.csv](data/catalog.csv): sample fashion dataset

---

## 8) Suggested GitHub Hosting Checklist

When publishing your final project repo:

1. Push your fork branch containing this project folder.
2. Add sample output screenshots of recommendations.
3. Add a short architecture diagram image.
4. Include a short demo GIF (optional but recommended).
5. Ensure README has setup steps and reproducible commands.

---

## 9) Future Enhancements

- Add user history and feedback loops for personalized reranking.
- Hybrid dense + sparse retrieval.
- Expose REST API with FastAPI.
- Add UI (Streamlit/Next.js) for interactive styling assistant.
- Add offline evaluation metrics (Recall@K, NDCG@K).
