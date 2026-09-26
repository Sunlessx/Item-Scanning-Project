# Visual Search

A standalone visual product-search microservice. Accept a product image, get back the closest-matching products in your catalog — in well under a millisecond at typical e-commerce catalog sizes.

Built for the "take a photo → find similar products" flow on a fast-fashion storefront.

## How it works

```
Client → POST /search (multipart image)
            │
            ▼
        FastAPI (port 8100)
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
 CLIP    FAISS    Typesense
(local) (in-proc) (read-only catalog source)
```

1. **CLIP** (ViT-B/32, runs locally — no API costs) generates a 512-dim embedding for the uploaded image.
2. **FAISS** (`IndexFlatIP`, in-process, ~40 MB for 20k products) does an exact cosine-similarity nearest-neighbour lookup.
3. The matching product metadata is returned. Indexed product metadata is sourced from **Typesense** (read-only — the indexer pulls the full catalog nightly).

No separate vector DB. No GPU required. The CLIP weights (~600 MB) are downloaded once on first run and cached in `~/.cache/huggingface/`.

## Why these choices

- **CLIP local, not an API** — image embedding is the hot path; paying per call doesn't scale. CPU inference is fast enough for the indexing job, and the API server reuses cached embeddings via Redis.
- **`IndexFlatIP`** — at ≤100k vectors, exact search is *faster* than IVF/HNSW because there's zero indexing overhead. Inner product on L2-normalised vectors == cosine similarity.
- **Typesense as catalog source, read-only** — schema decisions stay in the storefront's domain. This service never writes to it.
- **Atomic index swap** — the indexer writes to `.tmp` files and renames, so a long-running rebuild never corrupts a serving instance.

## API

### `GET /health`
```json
{ "status": "ok", "indexed_products": 21340 }
```

### `POST /search`
Multipart image upload. Returns top-k matching products.

```bash
curl -X POST http://localhost:8100/search \
  -F "image=@/path/to/product.jpg" \
  -F "top_k=5" | jq
```

```json
{
  "results": [
    {
      "product_id": "136904",
      "name": "Blue Strappy Heels",
      "img": "https://cdn.example.com/...",
      "selling_price": 999.0,
      "discount_price": 799.0,
      "score": 0.94
    }
  ],
  "count": 5
}
```

Supported formats: JPEG, PNG, WebP. Max upload: 10 MB. Magic-byte sniffing rejects mismatched extensions.

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/si88har1h/visual-search.git
cd visual-search
python3 -m venv venv && venv/bin/pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env with your Typesense host + read-only API key

# 3. Build a small test index
venv/bin/python scripts/index_catalog.py --limit 100

# 4. Start the API
venv/bin/uvicorn visual_search.main:app --host 127.0.0.1 --port 8100
```

### Indexing

| Command | What it does |
|---|---|
| `python scripts/index_catalog.py --limit 20 --dry-run` | Sanity check — embeds 20 products, writes nothing |
| `python scripts/index_catalog.py --limit 100` | Small index for testing the API |
| `python scripts/index_catalog.py` | Full catalog (≈30–90 min on CPU for 20k items) |

Index files land in `data/embeddings.index` (FAISS) and `data/metadata.pkl` (positionally-aligned product metadata). The API server loads both on startup.

## Docker

```bash
docker build -t visual-search .
docker run -p 8100:8100 --env-file .env -v $(pwd)/data:/app/data visual-search
```

The Dockerfile pre-downloads CLIP weights at build time to make cold starts instant.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TYPESENSE_HOST` | — | Typesense host (no default — must be set) |
| `TYPESENSE_PORT` | `443` | Typesense port |
| `TYPESENSE_PROTOCOL` | `https` | Typesense protocol |
| `TYPESENSE_API_KEY` | — | Read-only Typesense API key |
| `TYPESENSE_COLLECTION` | `products` | Collection name |
| `INDEX_PATH` | `data/embeddings.index` | FAISS index path |
| `METADATA_PATH` | `data/metadata.pkl` | Metadata sidecar path |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | HuggingFace model ID |
| `CLIP_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SEARCH_TOP_K` | `20` | Max results per search |
| `MAX_UPLOAD_BYTES` | `10485760` | Max upload size (10 MB) |
| `REDIS_URL` | `redis://localhost:6379` | Optional Redis for embedding/result caching |

Redis is optional — if unreachable, caching silently disables and search still works.

## Index rebuild strategy

The indexing script always rebuilds the full index (FAISS flat indices don't support deletion). Writes go to `.tmp` files and are atomically renamed; the running API server keeps serving until you restart it to pick up the new index.

For incremental updates later, swap `IndexFlatIP` for `IndexIDMap(IndexFlatIP)` to support add/remove by ID.

## License

MIT — see [LICENSE](./LICENSE).
