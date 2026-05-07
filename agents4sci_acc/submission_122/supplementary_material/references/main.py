"""

    Script to parse references in a generated text.

    Refer to `demo.py` for the prompt to use.

"""



import os
import argparse
import re
import requests
import urllib.request
import time
import json
import arxiv
from tqdm import tqdm
import logging

from typing import Optional

logger = logging.getLogger(__name__)



class Paper:
    def __init__(self, id, title, year, bibtex):
        self.id = id
        self.title = title
        self.year = year
        self.bibtex = bibtex


def get_bibtex_from_dxdoi(doi: str) -> str:
    """
        Retrieves the bibtex citation using doi.org API.
    """
    try:
        req = urllib.request.Request(f"http://dx.doi.org/{doi}")
        req.add_header('Accept', 'application/x-bibtex')
        with urllib.request.urlopen(req) as f:
            bibtex = f.read().decode()
    except:
        bibtex = None
    return bibtex


def get_from_semanticscholar(title: str, year: int, use_year: bool=False, retries: int=10) -> Optional[Paper]:
    """
        Queries the information about a paper from Semantic Scholar
    """
    for i in range(retries):
        params = {}
        params["query"] = title
        params["fields"] = "title,year,externalIds,venue,citationStyles"
        if use_year: params["year"] = year

        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/search/match",
            params = params
        )

        match resp.status_code:
            case 200:
                json_data = json.loads(resp.text)
                paper_data = json_data["data"][0]

                bibtex = None
                if "DOI" in paper_data["externalIds"]:
                    bibtex = get_bibtex_from_dxdoi(paper_data["externalIds"]["DOI"])
                if bibtex is None:
                    bibtex = paper_data["citationStyles"]["bibtex"]
                bibtex = re.sub(r"(@\w+{)([\w:/\.]+),", fr"\1 {paper_data['paperId']},", bibtex) # Use Semantic Scholar id as label
                
                return Paper(paper_data["paperId"], paper_data["title"], paper_data["year"], bibtex)
            case 404:
                return None
            case 429:
                time.sleep(i)
            case _:
                raise RuntimeError("API error")


def get_from_arxiv(title: str):
    """
        Queries the information about a paper from arxiv
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query = f"ti:{re.sub(r"[^A-Za-z ]", "", title)}", # Strip special characters to make query less awkward
        max_results = 1,
    )
    results = [*client.results(search)]
    if len(results) == 0:
        return None
    paper_data = results[0]

    paper_id = paper_data.get_short_id()
    if "v" in paper_id: paper_id = paper_id[:paper_id.index("v")] # Remove version from id
    doi = f"10.48550/arXiv.{paper_id}" # Warning: might change in future!
    bibtex = get_bibtex_from_dxdoi(doi)
    bibtex = re.sub(r"(@\w+{)([\w:\-/\.]+),", fr"\1 {paper_id},", bibtex) # Use arxiv id as label

    return Paper(paper_id, paper_data.title, paper_data.published, bibtex)


def get_paper(title: str, year: int) -> Optional[Paper]:
    """
        Retrieves the information about a paper
    """
    # Use Semantic Scholar
    paper = get_from_semanticscholar(title, year)
    # Fallback to arxiv
    if paper is None:
        paper = get_from_arxiv(title)

    if paper is None:
        logger.warning(f"{title} ({year}) not found")

    return paper


def get_all_citations(text: str) -> dict[str, Paper]:
    """
        Extracts all citations in the source text.
    """
    # Pattern matching from text
    to_process_citations = set()
    for cite in re.findall( r'\\cite{(.*?)}', text):
        citations = re.findall( r'(.*?)__(\d{4}?),?', cite)
        for c in citations:
            c = (c[0].strip(), c[1])
            to_process_citations.add(c)

    # Retrieve paper information
    all_citations = {}
    for c in tqdm(to_process_citations):
        title, year = c
        paper = get_paper(title, year)
        all_citations[f"{title}__{year}"] = paper
    return all_citations


def normalize_source(text: str, all_citations: dict[str, Paper]) -> str:
    """
        Normalizes the labels of the citations in the text
    """
    unk_idx = 0

    for id in all_citations.keys():
        paper = all_citations[id]
        if paper is not None:
            text = text.replace(id, paper.id)
        else:
            text = text.replace(id, f"unknown{unk_idx}")
            unk_idx += 1

    return text


def create_bib_file(all_citations: dict[str, Paper], out_file: str):
    """
        Creates the bibtex file for the given papers
    """
    open(out_file, "w").close()

    with open(out_file, "a") as f_out:
        for paper in all_citations.values():
            if paper is not None:
                f_out.write(f"{paper.bibtex}\n\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Experiment code generation.")
    parser.add_argument("--text-path", type=str, required=True, help="Path to generated text")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory where the output is saved")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    text = open(args.text_path, "r").read()
    all_citations = get_all_citations(text)
    text = normalize_source(text, all_citations)
    
    with open(os.path.join(args.out_dir, "parsed.tex"), "w") as f:
        f.write(text)
    create_bib_file(all_citations, os.path.join(args.out_dir, "references.bib"))
