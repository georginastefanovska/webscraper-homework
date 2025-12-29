import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class ScrapeConfig:
    site_root: str = "https://web-scraping.dev"
    products_pages_limit: int = 20
    testimonials_delay_s: float = 0.0
    reviews_delay_s: float = 0.1
    output_file: str = "webscraper.json"


def build_headers() -> Dict[str, str]:
    # Headers required by the site endpoints
    return {
        "x-secret-token": "secret123",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://web-scraping.dev/testimonials",
        "Content-Type": "application/json",
    }


def fetch_html(session: requests.Session, url: str, headers: Optional[Dict[str, str]] = None) -> str:
    resp = session.get(url, headers=headers)
    resp.raise_for_status()
    return resp.text


def parse_products_page(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(".product")
    results: List[Dict[str, str]] = []

    for card in cards:
        name_el = card.select_one("h3")
        price_el = card.select_one(".price")
        if not name_el or not price_el:
            continue

        results.append(
            {
                "name": name_el.get_text(strip=True),
                "price": price_el.get_text(strip=True),
            }
        )

    return results


def collect_products(session: requests.Session, cfg: ScrapeConfig) -> List[Dict[str, str]]:
    collected: List[Dict[str, str]] = []

    # Keep going until a page returns no products (or we hit a safety limit)
    for page_idx in range(1, cfg.products_pages_limit + 1):
        url = f"{cfg.site_root}/products?page={page_idx}"
        html = fetch_html(session, url)
        page_items = parse_products_page(html)

        if not page_items:
            break

        collected.extend(page_items)

    return collected


def parse_testimonials_block(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.select(".testimonial")
    out: List[Dict[str, str]] = []

    for block in blocks:
        text_el = block.select_one(".text")
        ident = block.select_one("identicon-svg")
        author = ident.get("username") if ident and ident.has_attr("username") else "User"

        if text_el:
            out.append({"text": text_el.get_text(strip=True), "author": author})

    return out


def collect_testimonials(session: requests.Session, cfg: ScrapeConfig, headers: Dict[str, str]) -> List[Dict[str, str]]:
    all_items: List[Dict[str, str]] = []
    page = 1

    while True:
        url = f"{cfg.site_root}/api/testimonials?page={page}"
        resp = session.get(url, headers=headers)

        # Stop if server says no / empty response
        if resp.status_code != 200 or not resp.text.strip():
            break

        page_items = parse_testimonials_block(resp.text)
        if not page_items:
            break

        all_items.extend(page_items)
        page += 1

        if cfg.testimonials_delay_s > 0:
            time.sleep(cfg.testimonials_delay_s)

    return all_items


def gql_reviews_query() -> str:
    return """
    query GetReviews($first: Int, $after: String) {
      reviews(first: $first, after: $after) {
        edges {
          node { text date }
          cursor
        }
        pageInfo { endCursor hasNextPage }
      }
    }
    """


def collect_reviews(session: requests.Session, cfg: ScrapeConfig, headers: Dict[str, str]) -> List[Dict[str, str]]:
    endpoint = f"{cfg.site_root}/api/graphql"
    after: Optional[str] = None
    keep_going = True

    all_reviews: List[Dict[str, str]] = []

    while keep_going:
        payload = {
            "query": gql_reviews_query(),
            "variables": {"first": 20, "after": after},
        }

        resp = session.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()

        body: Dict[str, Any] = resp.json()
        reviews_obj = (body.get("data") or {}).get("reviews") or {}

        edges = reviews_obj.get("edges") or []
        for edge in edges:
            node = edge.get("node") or {}
            all_reviews.append(
                {
                    "text": node.get("text", ""),
                    "date": node.get("date", "2023-01-01"),
                }
            )

        page_info = reviews_obj.get("pageInfo") or {}
        keep_going = bool(page_info.get("hasNextPage", False))
        after = page_info.get("endCursor", None)

        # Defensive stop: no edges means nothing more useful
        if not edges:
            break

        if cfg.reviews_delay_s > 0:
            time.sleep(cfg.reviews_delay_s)

    return all_reviews


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def run_scraper() -> None:
    cfg = ScrapeConfig()
    headers = build_headers()

    result: Dict[str, Any] = {"products": [], "testimonials": [], "reviews": []}

    with requests.Session() as session:
        print("1) Collecting product list...")
        result["products"] = collect_products(session, cfg)

        print("2) Collecting testimonials...")
        result["testimonials"] = collect_testimonials(session, cfg, headers)

        print("3) Collecting reviews (GraphQL)...")
        result["reviews"] = collect_reviews(session, cfg, headers)

    write_json(cfg.output_file, result)

    print(
        f"Saved to {cfg.output_file} | "
        f"Products: {len(result['products'])}, "
        f"Testimonials: {len(result['testimonials'])}, "
        f"Reviews: {len(result['reviews'])}"
    )


if __name__ == "__main__":
    run_scraper()
