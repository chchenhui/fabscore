import os
import time
from typing import Dict, List, Optional

import requests


UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{}"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{}.fasta"


def fetch_proteome_accessions(proteome_id: str, size: int = 500, max_pages: Optional[int] = None, delay: float = 0.2) -> List[str]:
    """Fetch all UniProt accessions for a proteome ID (e.g., UP000000625)."""
    params = {
        "query": f"proteome:{proteome_id}",
        "fields": "accession",
        "format": "json",
        "size": str(size),
    }
    accessions: List[str] = []
    url = UNIPROT_SEARCH
    pages = 0
    while url:
        r = requests.get(url, params=params if pages == 0 else None, timeout=30)
        if r.status_code != 200:
            break
        data = r.json()
        for rec in data.get("results", []):
            acc = rec.get("primaryAccession")
            if acc:
                accessions.append(acc)
        # pagination: look for 'Link' header with rel=next or 'next' in json
        next_url = None
        links = r.headers.get("Link", "")
        if links:
            # format: <url>; rel=next, <url>; rel=last
            for part in links.split(","):
                if 'rel="next"' in part:
                    start = part.find("<")
                    end = part.find(">", start + 1)
                    if start >= 0 and end > start:
                        next_url = part[start + 1 : end]
        if not next_url and data.get("next"):
            next_url = data["next"]
        url = next_url
        pages += 1
        if max_pages and pages >= max_pages:
            break
        if url:
            time.sleep(delay)
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for a in accessions:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def fetch_feature_flags(accession: str) -> Dict[str, bool]:
    """Fetch boolean flags for TRANSMEM, SIGNAL, COILED, REPEAT.
    Falls back to False on error.
    """
    flags = {
        'flag_TM': False,
        'flag_signal': False,
        'flag_coiled': False,
        'flag_repeat': False,
    }
    try:
        url = UNIPROT_ENTRY.format(accession)
        params = {
            'fields': 'ft_transmem,ft_signal,ft_coiled,ft_repeat',
            'format': 'json'
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return flags
        data = r.json()
        # recent API returns object; older may return results list
        feats = []
        if isinstance(data, dict) and 'features' in data:
            feats = data['features']
        elif isinstance(data, dict) and 'results' in data:
            # search-like response
            results = data.get('results', [])
            if results:
                feats = results[0].get('features', [])
        for f in feats:
            ftype = str(f.get('type', '')).lower()
            if 'transmem' in ftype:
                flags['flag_TM'] = True
            if 'signal' in ftype:
                flags['flag_signal'] = True
            if 'coiled' in ftype:
                flags['flag_coiled'] = True
            if 'repeat' in ftype:
                flags['flag_repeat'] = True
    except Exception:
        return flags
    return flags


def fetch_fasta(accession: str, timeout: int = 30) -> Optional[str]:
    try:
        r = requests.get(UNIPROT_FASTA.format(accession), timeout=timeout)
        if r.status_code == 200 and r.text.startswith('>'):
            return r.text
    except Exception:
        return None
    return None


def fetch_pdb_ids(accession: str, timeout: int = 30) -> List[str]:
    """Return list of PDB IDs linked to UniProt accession.

    Tries entry JSON first; falls back to search TSV with xref_pdb field.
    """
    # Primary: entry JSON
    try:
        r = requests.get(UNIPROT_ENTRY.format(accession), params={'format':'json'}, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            xrefs = []
            if isinstance(data, dict) and 'uniProtKBCrossReferences' in data:
                xrefs = data['uniProtKBCrossReferences']
            elif isinstance(data, dict) and 'results' in data:
                results = data.get('results', [])
                if results:
                    xrefs = results[0].get('uniProtKBCrossReferences', [])
            pdbs: List[str] = []
            for xr in xrefs:
                if xr.get('database') == 'PDB' and 'id' in xr:
                    pdbs.append(str(xr['id']).upper())
            if pdbs:
                # de-duplicate preserve order
                seen = set(); uniq: List[str] = []
                for p in pdbs:
                    if p not in seen:
                        seen.add(p); uniq.append(p)
                return uniq
    except Exception:
        pass
    # Fallback: TSV search with xref_pdb
    try:
        params = {
            'query': f'accession:{accession}',
            'fields': 'accession,xref_pdb',
            'format': 'tsv',
            'size': '1',
        }
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=timeout)
        if r.status_code != 200:
            return []
        lines = r.text.strip().splitlines()
        if len(lines) < 2:
            return []
        header = lines[0].split('\t')
        # Find PDB field column
        try:
            idx = header.index('Cross-reference (PDB)')
        except ValueError:
            # alternative header key
            idx = header.index('xref_pdb') if 'xref_pdb' in header else -1
        if idx < 0:
            return []
        vals = lines[1].split('\t')
        cell = vals[idx] if idx < len(vals) else ''
        if not cell:
            return []
        # Cell contains semicolon-separated PDB IDs
        pdbs = [p.strip().upper() for p in cell.split(';') if p.strip()]
        # de-duplicate
        seen = set(); uniq = []
        for p in pdbs:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return uniq
    except Exception:
        return []


def fetch_pdb_chain_ids(accession: str, timeout: int = 30) -> List[str]:
    """Return list of PDB chain IDs like 1ABC_A linked to UniProt accession.

    Parses UniProt entry cross-references to PDB and extracts chain letters
    from the 'chains' property (e.g., 'A=1-214; B=5-200').
    """
    try:
        r = requests.get(UNIPROT_ENTRY.format(accession), params={'format': 'json'}, timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
        xrefs = []
        if isinstance(data, dict) and 'uniProtKBCrossReferences' in data:
            xrefs = data['uniProtKBCrossReferences']
        elif isinstance(data, dict) and 'results' in data:
            results = data.get('results', [])
            if results:
                xrefs = results[0].get('uniProtKBCrossReferences', [])
        chains: List[str] = []
        for xr in xrefs:
            if xr.get('database') == 'PDB' and 'id' in xr:
                pdb = str(xr['id']).upper()
                props = xr.get('properties', [])
                chain_prop = None
                for p in props:
                    if p.get('key','').lower() in ('chains','chain'):
                        chain_prop = p.get('value','')
                        break
                if chain_prop:
                    # Formats observed:
                    #  - 'A=1-214; B=5-200'
                    #  - 'A/B/C/D' (slash-separated chains)
                    #  - 'A, B, C' (comma-separated)
                    # Normalize common separators to ';'
                    norm = chain_prop.replace('/', ';').replace(',', ';')
                    parts = [c.strip() for c in norm.split(';') if c.strip()]
                    for part in parts:
                        # For 'A=1-214' take 'A'; for 'A' keep as is
                        c = part.split('=')[0].strip()
                        # Keep simple alphanumeric chain IDs (1–2 chars typical)
                        if c:
                            chains.append(f"{pdb}_{c.upper()}")
                else:
                    # no chain information; at least include entry id
                    chains.append(pdb)
        # de-duplicate
        seen = set(); uniq: List[str] = []
        for cid in chains:
            if cid not in seen:
                seen.add(cid); uniq.append(cid)
        return uniq
    except Exception:
        return []
