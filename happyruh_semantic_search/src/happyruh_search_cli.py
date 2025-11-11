from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "semantic_collection"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL_NAME)
qdrant = QdrantClient(url=QDRANT_URL)

def search_qdrant(query, top_k=5):
    q_vec = embedder.encode([query])[0].tolist()
    results = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=q_vec, limit=top_k)
    if not results:
        print("No results found.")
        return

    print(f"\n🔍 Results for: '{query}'\n")
    for r in results:
        p = r.payload or {}
        name = p.get("product_name") or p.get("source_filename") or "(no name)"
        print(f"🧾 {name}")
        if p.get("type") == "product":
            print(f"💰 {p.get('price')}")
            print(f"🔗 {p.get('url')}")
        print(f"📈 Score: {round(r.score, 3)}\n")

if __name__ == "__main__":
    while True:
        q = input("\nEnter search query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        search_qdrant(q)
