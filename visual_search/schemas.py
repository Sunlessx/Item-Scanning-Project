from pydantic import BaseModel
from typing import Optional


class ProductMatch(BaseModel):
    product_id: str
    name: str
    img: str
    selling_price: float
    discount_price: Optional[float] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    results: list[ProductMatch]
    count: int
