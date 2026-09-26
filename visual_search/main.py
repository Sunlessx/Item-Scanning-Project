from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import settings
from .embedder import embed_image_bytes
from .index_store import IndexStore
from .redis_cache import cache_get, cache_set, hash_bytes
from .schemas import ProductMatch, SearchResponse

SEARCH_TTL = 3600

store = IndexStore(settings.index_path, settings.metadata_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if store.index_path.exists() and store.metadata_path.exists():
        store.load()
    yield


app = FastAPI(title="Visual Search", version="1.0.0", lifespan=lifespan)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",
}


def detect_image_type(data: bytes) -> str | None:
    for sig, mime in IMAGE_SIGNATURES.items():
        if data[:len(sig)] == sig:
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return None


@app.get("/health")
def health():
    return {"status": "ok", "indexed_products": store.total}


@app.post("/search", response_model=SearchResponse)
async def search(image: UploadFile = File(...), top_k: int = 10):
    data = await image.read()

    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    mime = detect_image_type(data) or image.content_type
    if mime not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, WebP accepted")

    if not store.is_loaded:
        raise HTTPException(status_code=503, detail="Index not ready — run scripts/index_catalog.py first")

    k = min(top_k, settings.search_top_k)
    embed_h = hash_bytes(data)
    search_key = f"img:search:{embed_h}:{k}"
    hits = cache_get(search_key)
    if hits is None:
        embedding = embed_image_bytes(data)
        hits = store.search(embedding, top_k=k)
        cache_set(search_key, hits, SEARCH_TTL)

    results = [
        ProductMatch(
            product_id=str(h.get("product_id", h.get("id", ""))),
            name=h.get("name", ""),
            img=h.get("img", ""),
            selling_price=float(h.get("selling_price", 0)),
            discount_price=float(h["discount_price"]) if h.get("discount_price") else None,
            score=h.get("score"),
        )
        for h in hits
    ]
    return SearchResponse(results=results, count=len(results))
