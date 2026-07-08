import os
import re
import time
import hashlib
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.mrfog.online/"
OUTPUT_DIR = os.path.join("docs", "mrfog_online")
CATALOG_SUMMARY_PATH = os.path.join("docs", "mrfog_online_catalog_summary.txt")

MAX_PAGES = 250
DELAY_SECONDS = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MRFOG-AI-Assistant/1.0; +local-rag-practice)"
}

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".zip", ".mp4", ".mov", ".avi", ".css", ".js", ".ico"
)

SKIP_PATH_PARTS = [
    "/cart",
    "/checkout",
    "/account",
    "/login",
    "/register",
    "/password",
    "/search",
    "/policies",
]


def clean_url(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/") + "/"


def is_valid_internal_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.scheme not in ["http", "https"]:
        return False

    if parsed.netloc not in ["www.mrfog.online", "mrfog.online"]:
        return False

    lower_url = url.lower()

    if any(lower_url.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False

    if any(part in lower_url for part in SKIP_PATH_PARTS):
        return False

    return True


def slugify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        path = "home"

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    short_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:6]

    return f"mrfog_online_{slug}_{short_hash}.txt"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_lines(soup: BeautifulSoup):
    for tag in soup(["style", "noscript", "iframe"]):
        tag.decompose()

    raw_lines = soup.get_text("\n", strip=True).splitlines()

    lines = []
    seen = set()

    for line in raw_lines:
        line = clean_text(line)

        if not line:
            continue

        if len(line) < 2:
            continue

        if line in seen:
            continue

        seen.add(line)
        lines.append(line)

    return lines


def looks_like_product_title(line: str) -> bool:
    text = line.lower()

    product_words = [
        "mr fog",
        "doozy",
        "nova",
        "aura",
        "switch",
        "max air",
        "max pro",
        "drt",
        "pod kit",
        "puffs",
        "disposable vape",
        "e-liquid",
        "steezy",
        "vape",
        "pods pack",
        "charger cable",
        "powerbank",
    ]

    bad_words = [
        "add to cart",
        "view more",
        "shop now",
        "home",
        "log in",
        "create an account",
        "warning",
        "free shipping",
        "secure online payment",
    ]

    if any(bad in text for bad in bad_words):
        return False

    return any(word in text for word in product_words)


def extract_price_near(lines, start_index):
    price_pattern = re.compile(r"\$\d+(?:\.\d{2})?(?:\s+\$\d+(?:\.\d{2})?)?")

    for j in range(start_index + 1, min(start_index + 8, len(lines))):
        match = price_pattern.search(lines[j])
        if match:
            return match.group(0)

    return ""


def infer_category(title: str) -> str:
    text = title.lower()

    if "aura" in text:
        return "AURA / AURA Splash"
    if "nova" in text:
        return "NOVA"
    if "switch pod" in text:
        return "SWITCH POD"
    if "switch 15000" in text:
        return "SWITCH 15000"
    if "switch 5500" in text:
        return "SWITCH 5500"
    if "max air 8500" in text:
        return "MAX AIR 8500"
    if "max air 3000" in text:
        return "MAX AIR 3000"
    if "max pro" in text:
        return "MAX PRO"
    if "max 1000" in text:
        return "MAX 1000"
    if "drt" in text:
        return "DRT Open System"
    if "e-liquid" in text or "steezy" in text and "ml" in text:
        return "E-Liquid / Vapor Fluid"
    if "doozy" in text:
        return "Powered By MR FOG / DOOZY"
    if "pod" in text:
        return "Pods / Accessories"

    return "Other"


def extract_products_from_lines(lines, source_url):
    products = []

    for i, line in enumerate(lines):
        if looks_like_product_title(line):
            price = extract_price_near(lines, i)
            category = infer_category(line)

            products.append({
                "title": line,
                "price": price,
                "category": category,
                "source_url": source_url
            })

    return products


def extract_page_text_and_links(url: str, html: str):
    soup = BeautifulSoup(html, "lxml")

    title = "Untitled"
    title_tag = soup.find("title")
    if title_tag:
        title = clean_text(title_tag.get_text())

    lines = extract_page_lines(soup)
    products = extract_products_from_lines(lines, url)

    output_lines = []
    output_lines.append(f"Source URL: {url}")
    output_lines.append(f"Page Title: {title}")
    output_lines.append("Document Type: MR FOG Online Store Page")
    output_lines.append("")

    output_lines.append("PAGE CONTENT")
    output_lines.extend(lines)

    if products:
        output_lines.append("")
        output_lines.append("EXTRACTED PRODUCT INFORMATION")
        for product in products:
            output_lines.append(f"Product: {product['title']}")
            output_lines.append(f"Category: {product['category']}")
            output_lines.append(f"Price: {product['price']}")
            output_lines.append(f"Source: {product['source_url']}")
            output_lines.append("")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = clean_url(urljoin(url, href))

        if is_valid_internal_url(full_url):
            links.append(full_url)

    return "\n".join(output_lines), sorted(set(links)), products


def write_catalog_summary(all_products):
    unique = {}

    for product in all_products:
        key = product["title"].strip().lower()

        if key not in unique:
            unique[key] = product

    products = list(unique.values())

    by_category = {}

    for product in products:
        by_category.setdefault(product["category"], []).append(product)

    lines = []
    lines.append("Source URL: https://www.mrfog.online/")
    lines.append("Document Type: MR FOG Online Store Catalog Summary")
    lines.append("Purpose: Product, flavor, category, and pricing summary for MRFOG AI ASSISTANT.")
    lines.append("")
    lines.append("IMPORTANT NOTE")
    lines.append("This catalog summary is generated from the public MR FOG online store pages.")
    lines.append("Prices and availability can change, so refresh the scraper before relying on current pricing.")
    lines.append("")
    lines.append(f"Total unique product/title entries extracted: {len(products)}")
    lines.append("")
    lines.append("PRODUCT CATEGORIES FOUND")

    for category in sorted(by_category.keys()):
        lines.append(f"- {category}: {len(by_category[category])} items")

    lines.append("")
    lines.append("PRODUCT LIST BY CATEGORY")

    for category in sorted(by_category.keys()):
        lines.append("")
        lines.append(f"## {category}")

        for product in sorted(by_category[category], key=lambda x: x["title"]):
            price_text = product["price"] if product["price"] else "Price not found"
            lines.append(f"- {product['title']} | Price: {price_text} | Source: {product['source_url']}")

    with open(CATALOG_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Catalog summary saved: {CATALOG_SUMMARY_PATH}")
    print(f"Unique products extracted: {len(products)}")


def crawl_site():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_url = clean_url(BASE_URL)

    queue = [start_url]
    visited = set()
    all_products = []

    seed_urls = [
        "https://www.mrfog.online/",
        "https://www.mrfog.online/collections/mr-fog-product",
        "https://www.mrfog.online/collections/disposable-pods",
        "https://www.mrfog.online/collections/e-liquid",
        "https://www.mrfog.online/collections/closed-pod-system",
        "https://www.mrfog.online/collections/open-system",
        "https://www.mrfog.online/collections/mr-fog-drt",
        "https://www.mrfog.online/collections/mr-fog-aura",
        "https://www.mrfog.online/collections/mr-fog-aura-splash-series",
        "https://www.mrfog.online/collections/mr-fog-nova-original",
        "https://www.mrfog.online/collections/mr-fog-switch-pod-kit",
        "https://www.mrfog.online/collections/mr-fog-switch-pod-pods",
        "https://www.mrfog.online/collections/powered-by-mr-fog",
    ]

    for seed in seed_urls:
        seed = clean_url(seed)
        if seed not in queue:
            queue.append(seed)

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)

        if url in visited:
            continue

        print(f"Crawling: {url}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=25)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed: {url} | {e}")
            continue

        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type:
            continue

        visited.add(url)

        page_text, links, products = extract_page_text_and_links(url, response.text)

        all_products.extend(products)

        if len(page_text) > 300:
            filename = slugify_url(url)
            output_path = os.path.join(OUTPUT_DIR, filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(page_text)

            print(f"Saved: {output_path}")

        for link in links:
            if link not in visited and link not in queue:
                queue.append(link)

        time.sleep(DELAY_SECONDS)

    write_catalog_summary(all_products)

    print("")
    print(f"Done. Pages visited: {len(visited)}")
    print(f"Docs folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    crawl_site()