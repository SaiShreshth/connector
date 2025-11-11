# app/api/v1/routes/classification_route.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from app.ai.intent_classification.hybrid_classifier import HybridIntentClassifier

# Initialize FastAPI router
router = APIRouter()
classifier = HybridIntentClassifier()

# ----- SCHEMA DEFINITIONS -----
class ClassifyRequest(BaseModel):
    text: str

class PriceRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    currency: Optional[str] = "INR"

class Entities(BaseModel):
    product_name: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    price_range: Optional[PriceRange] = None
    category: Optional[str] = None

class SearchResult(BaseModel):
    product_name: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    color: Optional[str] = None

class ClassifyResponse(BaseModel):
    action_code: str
    confidence: float
    entities: Optional[Entities] = None
    search_results: Optional[List[SearchResult]] = None
    original_text: str


# ----- ENDPOINT -----
@router.post("/classify", response_model=ClassifyResponse, summary="Classify user query into intent + entities")
async def classify_text(req: ClassifyRequest):
    """
    Classifies a user query into an intent (like SEARCH, CANCEL)
    and extracts relevant entities (brand, color, price, etc.)
    """
    return classifier.classify(req.text)
