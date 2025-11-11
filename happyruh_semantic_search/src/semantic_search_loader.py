import os
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from db import ensure_tables, fetch_all_raw
from data_cleaner import clean_text, parse_price_value

BASE_URL = "https://happyruh.com"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "semantic_collection")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

embedder = SentenceTransformer(EMBED_MODEL_NAME)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    timeout=180.0
)

def ensure_collection_exists():
    names = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in names:
        print(f"🆕 Creating collection '{QDRANT_COLLECTION}'")
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
    else:
        print(f"✅ Collection '{QDRANT_COLLECTION}' exists — preserving data.")

def build_clean_text_and_payload(raw_row):
    pjson = raw_row["product_json"] or {}
    handle = pjson.get("handle") or raw_row.get("handle")
    url = raw_row.get("url") or (urljoin(BASE_URL, f"/products/{handle}") if handle else None)
    title = pjson.get("title") or raw_row.get("title") or ""
    body_html = pjson.get("body_html") or ""
    description_clean = clean_text(body_html)  # CLEANING ONLY HERE
    image = pjson.get("image", {}).get("src") if pjson.get("image") else None
    product_type = pjson.get("product_type") or None
    variants = pjson.get("variants") or []
    price_value = parse_price_value(variants[0].get("price")) if variants else None

    text = (
        f"Product Name: {title}. "
        f"Price: {price_value} INR. "
        f"Description: {description_clean}. "
        f"Category: {product_type}."
    )
    payload = {
        "type": "product",
        "source": "happyruh_raw_shopify",
        "text": text,
        "product_name": title,
        "price": f"₹{price_value:.2f}" if price_value is not None else None,
        "price_value": price_value,
        "url": url,
        "image": image,
        "category": product_type,
    }
    return text, payload

def store_products_in_qdrant_from_postgres():
    ensure_tables()
    ensure_collection_exists()

    rows = fetch_all_raw()
    if not rows:
        print("⚠️ No RAW rows in Postgres.")
        return

    texts, payloads, ids = [], [], []
    for rr in rows:
        t, pl = build_clean_text_and_payload(rr)
        texts.append(t); payloads.append(pl); ids.append(int(rr["product_id"]))

    print("🧠 Generating embeddings…")
    vectors = embedder.encode(texts)

    points = [
        models.PointStruct(
            id=ids[i],
            vector=vectors[i].tolist(),
            payload=payloads[i],
        )
        for i in range(len(texts))
    ]

    print(f"⬆️ Uploading {len(points)} products to Qdrant…")
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print("✅ Upload complete (cleaning applied only in Qdrant payload/vector).")

if __name__ == "__main__":
    store_products_in_qdrant_from_postgres()
    print("🚀 Qdrant synced from RAW Postgres.")
