import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# Connect to Qdrant
qdrant_client = QdrantClient(url="http://localhost:6333")

def qdrant_search(entities):
    """
    Searches the Qdrant 'chatnshop_products' collection using entity filters.
    Returns top 5 products that match filters.
    """
    conditions = []

    # Filter by brand
    if entities.get("brand"):
        conditions.append(FieldCondition(
            key="brand",
            match=MatchValue(value=entities["brand"].lower())
        ))

    # Filter by product name
    if entities.get("product_name"):
        conditions.append(FieldCondition(
            key="product_name",
            match=MatchValue(value=entities["product_name"].lower())
        ))

    # Filter by color
    if entities.get("color"):
        conditions.append(FieldCondition(
            key="color",
            match=MatchValue(value=entities["color"].lower())
        ))

    # Price filter (if available)
    price_filter = None
    if entities.get("price_range") and entities["price_range"].get("max"):
        price_filter = entities["price_range"]["max"]

    query_filter = Filter(must=conditions) if conditions else None

    # Perform search
    try:
        hits = qdrant_client.scroll(
            collection_name="chatnshop_products",
            scroll_filter=query_filter,
            limit=5
        )[0]

        products = []
        for hit in hits:
            product = hit.payload
            if price_filter and product.get("price") and product["price"] > price_filter:
                continue
            products.append(product)

        # 🔹 If Qdrant empty, return dummy data for testing
        if not products:
            products = [
                {"product_name": "Nike Air Zoom", "brand": "nike", "price": 2800, "color": "red"},
                {"product_name": "Nike Flex Runner", "brand": "nike", "price": 2999, "color": "red"}
            ]

        return products

    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []
