# Verifier Agent: Cross-references drug-symptom associations with FDA FAERS database for validation

import requests
import json
import re
from typing import Dict, List, Tuple, Optional, Iterable, Set
from collections import Counter
import math

def get_variants_and_terms(extractor, drug_name, receivedate_range=None):
    variants = extractor.get_product_name_variants(drug_name)
    faers_terms = extractor.get_faers_term_counts(
        variants,
        limit=200,
        min_count=1,
        receivedate_range=receivedate_range
    )
    return variants, faers_terms

def get_side_effect_score(extractor, variants, faers_terms, drug_name: str, side_effect: str,
                          receivedate_range=None, label_boost: float = 2.0):
    count = 0
    for t, c in faers_terms:
        if t.lower() == side_effect.lower():
            count = c
            break

    score = math.log1p(count)

    labeling = extractor.search_drug_labeling(drug_name)
    adverse_text = ' '.join(
        ' '.join(r.get('adverse_reactions', []))
        for r in labeling.get('results', [])
    )

    if re.search(rf'\b{re.escape(side_effect.lower())}\b', adverse_text.lower()):
        score += label_boost

    return {
        "drug": drug_name,
        "side_effect": side_effect,
        "count": count,
        "score": round(score, 2)
    }

class FDAAdverseReactionExtractor:
    def __init__(self):
        self.label_url = "https://api.fda.gov/drug/label.json"
        self.faers_url = "https://api.fda.gov/drug/event.json"

    def search_drug_labeling(self, drug_name: str, target_year: Optional[int] = None, limit: int = 50) -> Dict:
        search_query = f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"'
        params = {
            'search': search_query,
            'limit': limit
        }
        try:
            response = requests.get(self.label_url, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if target_year:
                results = self._filter_by_year(results, target_year)
            data["results"] = results
            return data

        except:
            return