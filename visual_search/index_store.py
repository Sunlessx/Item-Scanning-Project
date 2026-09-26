from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class IndexStore:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_index = self.index_path.with_suffix(".index.tmp")
        tmp_meta = self.metadata_path.with_suffix(".pkl.tmp")
        faiss.write_index(self.index, str(tmp_index))
        with open(tmp_meta, "wb") as f:
            pickle.dump(self.metadata, f)
        tmp_index.rename(self.index_path)
        tmp_meta.rename(self.metadata_path)

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query: np.ndarray, top_k: int = 20) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({**self.metadata[idx], "score": float(score)})
        return results

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0
