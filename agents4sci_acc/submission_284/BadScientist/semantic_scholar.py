import os
import requests
import time
from typing import List, Dict

S2_API_KEY = os.getenv("S2_API_KEY")


def search(query: str, limit: int = 10) -> List[Dict]:
    """Search Semantic Scholar with retry mechanism"""
    max_retries = 20
    base_delay = 1.0  # 1 second
    max_delay = 64.0  # 64 seconds

    for attempt in range(max_retries):
        try:
            rsp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers={"X-API-KEY": S2_API_KEY} if S2_API_KEY else {},
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
                },
                timeout=30,
            )
            rsp.raise_for_status()
            data = rsp.json()
            return data.get("data", [])
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with max delay cap
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"Semantic Scholar API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                print(f"Semantic Scholar API call failed after {max_retries} attempts: {e}")
                raise e

    # This should never be reached, but just in case
    return []


def collect_bibtex(
    queries: List[str],
    per_query: int = 8,
    max_total: int = 30,
    enable_topup: bool = True,
) -> List[str]:
    refs: List[str] = []
    # Primary pass
    for q in queries or []:
        try:
            papers = search(q, limit=per_query)
            for p in papers:
                cs = p.get("citationStyles") or {}
                bib = cs.get("bibtex")
                if not bib:
                    title = p.get("title", "Unknown")
                    authors = ", ".join(
                        a.get("name", "?") for a in p.get("authors", [])
                    )
                    venue = p.get("venue", "")
                    year = p.get("year", "")
                    bib = f"{authors}. {title}. {venue}, {year}."
                refs.append(bib)
        except Exception as e:
            print(f"Failed to collect references for query '{q}' after all retries: {e}")
            continue

    # Deduplicate
    uniq, seen = [], set()
    for r in refs:
        if r not in seen:
            uniq.append(r)
            seen.add(r)

    # Top-up with query variants if still under cap
    if enable_topup and len(uniq) < max_total:
        variants = [" review", " survey", " tutorial"]
        for suffix in variants:
            for q in queries or []:
                if len(uniq) >= max_total:
                    break
                try:
                    papers = search(q + suffix, limit=max(2, per_query // 2))
                    for p in papers:
                        cs = p.get("citationStyles") or {}
                        bib = cs.get("bibtex")
                        if not bib:
                            title = p.get("title", "Unknown")
                            authors = ", ".join(
                                a.get("name", "?") for a in p.get("authors", [])
                            )
                            venue = p.get("venue", "")
                            year = p.get("year", "")
                            bib = f"{authors}. {title}. {venue}, {year}."
                        if bib not in seen:
                            uniq.append(bib)
                            seen.add(bib)
                        if len(uniq) >= max_total:
                            break
                except Exception as e:
                    print(f"Failed to collect additional references for query '{q + suffix}' after all retries: {e}")
                    continue

    return uniq[:max_total]
