import os
import json
import time
import requests
from urllib.parse import urljoin
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from db import ensure_tables, upsert_raw

BASE_URL = "https://happyruh.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# --- Keys / config from env ---
SF_TOKEN = os.getenv("SHOPIFY_STOREFRONT_TOKEN")
SF_DOMAIN = os.getenv("SHOPIFY_DOMAIN", "happyruh.com")
SF_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2024-07")

ADMIN_KEY = os.getenv("SHOPIFY_ADMIN_API_KEY")
ADMIN_PASS = os.getenv("SHOPIFY_ADMIN_PASSWORD")
ADMIN_SHOP = os.getenv("SHOPIFY_SHOP")  # e.g., 'happyruh' (no suffix) or 'yourshop.myshopify.com'

def save_audit_files(raw_products):
    path_json = os.path.join(DATA_DIR, "happyruh_products.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(raw_products, f, indent=2, ensure_ascii=False)
    print(f"💾 RAW JSON audit saved to {path_json}")

    path_pdf = os.path.join(DATA_DIR, "happyruh_products.pdf")
    pdf = canvas.Canvas(path_pdf, pagesize=A4)
    w, h = A4
    y = h - 60
    pdf.setTitle("HappyRuH Product Catalog (RAW snapshot)")
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(120, y, "HappyRuH Product Catalog (RAW snapshot)")
    y -= 30
    pdf.setFont("Helvetica", 9)

    if not raw_products:
        pdf.drawString(40, y, "No products found.")
        pdf.save(); print(f"📄 PDF saved to {path_pdf}")
        return path_json, path_pdf

    for rp in raw_products:
        if y < 80:
            pdf.showPage(); pdf.setFont("Helvetica", 9); y = h - 60
        title = rp.get("title") or ""
        handle = rp.get("handle") or ""
        url = urljoin(BASE_URL, f"/products/{handle}") if handle else ""
        price = None
        variants = rp.get("variants") or []
        if variants and variants[0].get("price"):
            price = variants[0]["price"]
        pdf.drawString(40, y, f"Name: {title}"); y -= 14
        if price: pdf.drawString(40, y, f"Price(raw): {price}"); y -= 14
        if url:   pdf.drawString(40, y, f"URL: {url}"); y -= 14
        y -= 6

    pdf.save()
    print(f"📄 PDF saved to {path_pdf}")
    return path_json, path_pdf

# ---------- Path A: Shopify Storefront GraphQL ----------
def fetch_products_storefront_graphql():
    if not SF_TOKEN:
        return None
    url = f"https://{SF_DOMAIN}/api/{SF_API_VERSION}/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": SF_TOKEN,
    }
    query = """
    query getProducts($cursor: String) {
      products(first: 250, after: $cursor) {
        edges {
          cursor
          node {
            id
            handle
            title
            descriptionHtml
            productType
            images(first: 1) { edges { node { src: url } } }
            variants(first: 1) { edges { node { price { amount currencyCode } } } }
          }
        }
        pageInfo { hasNextPage }
      }
    }
    """
    variables = {"cursor": None}
    out = []
    while True:
        resp = requests.post(url, headers=headers, json={"query": query, "variables": variables}, timeout=45)
        if resp.status_code != 200:
            print("Storefront GraphQL failed:", resp.status_code, resp.text[:300])
            return out or None
        data = resp.json()
        edges = (((data or {}).get("data") or {}).get("products") or {}).get("edges") or []
        for e in edges:
            n = e["node"]
            # normalize to your schema
            price = None
            v_edges = (((n.get("variants") or {}).get("edges")) or [])
            if v_edges:
                p = v_edges[0]["node"]["price"]
                if p and p.get("amount"):
                    price = p["amount"]
            img = None
            i_edges = (((n.get("images") or {}).get("edges")) or [])
            if i_edges:
                img = i_edges[0]["node"].get("src")

            out.append({
                "id": int(abs(hash(n["id"])) % 10**12),  # Storefront IDs are global IDs; create a stable numeric
                "handle": n.get("handle"),
                "title": n.get("title"),
                "body_html": n.get("descriptionHtml"),
                "product_type": n.get("productType"),
                "image": {"src": img} if img else None,
                "variants": [{"price": price}] if price else [],
            })
        page_info = (((data or {}).get("data") or {}).get("products") or {}).get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        variables["cursor"] = edges[-1]["cursor"]
        time.sleep(0.25)
    return out

# ---------- Path B: Shopify Admin REST ----------
def build_admin_base():
    if not (ADMIN_KEY and ADMIN_PASS and ADMIN_SHOP):
        return None
    shop = ADMIN_SHOP
    if not shop.endswith(".myshopify.com"):
        shop = f"{shop}.myshopify.com"
    return f"https://{ADMIN_KEY}:{ADMIN_PASS}@{shop}/admin/api/{SF_API_VERSION}"

def fetch_products_admin_rest(limit=250, max_pages=40):
    base = build_admin_base()
    if not base:
        return None
    out, page = [], 1
    while page <= max_pages:
        url = f"{base}/products.json"
        params = {"limit": str(limit), "page": str(page)}
        r = requests.get(url, params=params, timeout=45)
        if r.status_code != 200:
            print("Admin REST failed:", r.status_code, r.text[:300])
            return out or None
        data = r.json() or {}
        products = data.get("products") or []
        out.extend(products)
        if len(products) < limit:
            break
        page += 1
        time.sleep(0.25)
    return out

# ---------- Path C: Public JSON (fallback) ----------
def get_json(url, params=None, retry=3, sleep=0.8):
    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    pass
            if r.status_code in (429, 503):
                time.sleep(sleep * (i + 1))
                continue
        except Exception:
            time.sleep(sleep * (i + 1))
    return None

def fetch_products_public_json(limit=250, max_pages=40):
    all_products, page = [], 1
    while page <= max_pages:
        url = urljoin(BASE_URL, "/products.json")
        data = get_json(url, params={"limit": str(limit), "page": str(page)})
        if not data or not isinstance(data, dict) or not data.get("products"):
            break
        products = data["products"]
        all_products.extend(products)
        if len(products) < limit:
            break
        page += 1
        time.sleep(0.25)
    return all_products

# ---------- Main runner ----------
if __name__ == "__main__":
    ensure_tables()
    print("🛒 Fetching products with key-aware strategy …")

    raw_products = None

    # 1) Try Storefront GraphQL
    if SF_TOKEN:
        print("→ Using Shopify Storefront GraphQL")
        raw_products = fetch_products_storefront_graphql()

    # 2) Else try Admin REST
    if (not raw_products) and ADMIN_KEY and ADMIN_PASS and ADMIN_SHOP:
        print("→ Using Shopify Admin REST")
        raw_products = fetch_products_admin_rest()

    # 3) Else fallback to public JSON
    if not raw_products:
        print("→ Falling back to public JSON endpoints")
        raw_products = fetch_products_public_json()

    # Upsert RAW into Postgres
    count = 0
    for rp in raw_products or []:
        pid = rp.get("id")
        handle = rp.get("handle")
        title = rp.get("title") or ""
        url = urljoin(BASE_URL, f"/products/{handle}") if handle else None
        upsert_raw({"id": pid, "handle": handle, "title": title, "url": url, "json": rp})
        count += 1

    print(f"✅ RAW products upserted to Postgres: {count}")
    save_audit_files(raw_products or [])
    print("🎉 RAW stored in Postgres; audit JSON/PDF created.")
