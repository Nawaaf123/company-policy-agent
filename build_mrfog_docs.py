import os
import re
import time
import hashlib
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.mrfog.com/"
OUTPUT_DIR = os.path.join("docs", "mrfog")

MAX_PAGES = 150
DELAY_SECONDS = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CompanyPolicyAgent/1.0; +local-practice-rag)"
}

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf",
    ".zip", ".mp4", ".mov", ".avi", ".css", ".js", ".ico"
)

SKIP_PATH_PARTS = [
    "/wp-admin",
    "/wp-login",
    "/cart",
    "/checkout",
    "/my-account",
    "/feed",
    "/comments",
]


def clean_url(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/") + "/"


def is_valid_internal_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.scheme not in ["http", "https"]:
        return False

    if parsed.netloc not in ["www.mrfog.com", "mrfog.com"]:
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

    return f"mrfog_{slug}_{short_hash}.txt"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_text(url: str, html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    title = clean_text(soup.get_text(" ", strip=True)[:100])

    title_tag = soup.find("title")
    if title_tag:
        title = clean_text(title_tag.get_text())

    lines = []
    lines.append(f"Source URL: {url}")
    lines.append(f"Page Title: {title}")
    lines.append("")

    # Extract headings, paragraphs, list items, table cells, buttons, and image alt text
    useful_tags = soup.find_all([
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "td", "th", "span", "strong"
    ])

    seen = set()

    for tag in useful_tags:
        text = clean_text(tag.get_text(" ", strip=True))

        if not text:
            continue

        if len(text) < 3:
            continue

        if text in seen:
            continue

        seen.add(text)
        lines.append(text)

    # Add image alt/title text because many product pages store flavor names in images
    for img in soup.find_all("img"):
        alt = clean_text(img.get("alt", ""))
        title_attr = clean_text(img.get("title", ""))

        for img_text in [alt, title_attr]:
            if img_text and img_text not in seen and len(img_text) > 2:
                seen.add(img_text)
                lines.append(f"Image text: {img_text}")

    # Extract internal links
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = clean_url(urljoin(url, href))

        if is_valid_internal_url(full_url):
            links.append(full_url)

    return "\n".join(lines), sorted(set(links))


def crawl_site():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_url = clean_url(BASE_URL)

    queue = [start_url]
    visited = set()

    # Important seed pages from public MR FOG navigation/product pages
    seed_urls = [
        "https://www.mrfog.com/",
        "https://www.mrfog.com/nova-series/",
        "https://www.mrfog.com/mr-fog-nova/",
        "https://www.mrfog.com/mr-fog-drt/",
        "https://www.mrfog.com/mr-fog-switch-pod/",
        "https://www.mrfog.com/switch-5500/",
        "https://www.mrfog.com/faq/",
        "https://www.mrfog.com/contact/",
        "https://www.mrfog.com/distributor-application-form/",
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
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed: {url} | {e}")
            continue

        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type:
            continue

        visited.add(url)

        page_text, links = extract_page_text(url, response.text)

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

    print("")
    print(f"Done. Pages saved: {len(visited)}")
    print(f"Docs folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    crawl_site()