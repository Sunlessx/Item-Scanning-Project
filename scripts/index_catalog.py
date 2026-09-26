#!/usr/bin/env python3
"""
Batch indexing pipeline — fetches all active products from Typesense,
generates CLIP embeddings, and saves a FAISS index + metadata to disk.

Usage:
    python scripts/index_catalog.py
    python scripts/index_catalog.py --limit 50      # testing
    python scripts/index_catalog.py --dry-run       # no files written
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_search.config import settings
from visual_search.embedder import embed_image_url
from visual_search.index_store import IndexStore
from visual_search.typesense_client import fetch_all_products, get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("indexer")


def run(limit: int = 0, dry_run: bool = False) -> None:
    client = get_client()
    embeddings = []
    metadata = []
    errors = 0

    log.info("Fetching products from Typesense...")
    for i, doc in enumerate(fetch_all_products(client)):
        if limit and i >= limit:
            break

        img_url = doc.get("img", "")
        if not img_url:
            log.warning(f"No image for product {doc.get('id')} — skipping")
            errors += 1
            continue

        try:
            vec = embed_image_url(img_url)
            embeddings.append(vec)
            metadata.append({
                "id": doc.get("id", ""),
                "product_id": str(doc.get("product_id", doc.get("id", ""))),
                "name": doc.get("name", ""),
                "img": img_url,
                "selling_price": float(doc.get("selling_price", 0)),
                "discount_price": float(doc["discount_price"]) if doc.get("discount_price") else None,
            })

            if (i + 1) % 100 == 0:
                log.info(f"Embedded {i + 1} products (errors={errors})")

            time.sleep(0.05)

        except Exception as e:
            log.error(f"Failed on product {doc.get('id')}: {e}")
            errors += 1

    if not embeddings:
        log.error("No embeddings generated — nothing to save")
        return

    log.info(f"Building FAISS index with {len(embeddings)} vectors...")
    arr = np.stack(embeddings)
    store = IndexStore(settings.index_path, settings.metadata_path)
    store.build(arr, metadata)

    if dry_run:
        log.info(f"Dry run — skipping save. Would have saved {len(embeddings)} vectors (errors={errors})")
    else:
        store.save()
        log.info(f"Saved index to {settings.index_path} and metadata to {settings.metadata_path} (errors={errors})")
        try:
            from visual_search.redis_cache import cache_del
            deleted = cache_del("img:search:*")
            log.info(f"Invalidated {deleted} cached search results")
        except Exception as e:
            log.warning(f"Cache invalidation skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max products to index (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Skip saving index files")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)
