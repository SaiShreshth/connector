from pydantic import BaseModel
from typing import Optional

class PriceRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    currency: Optional[str] = None

class Entities(BaseModel):
    product_type: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    price_range: Optional[PriceRange] = None
