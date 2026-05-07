import os
import subprocess
from typing import Dict, List, Tuple


def run(cmd: List[str], cwd: str | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stdout}")


def mmseqs_cluster(fasta_path: str, out_dir: str, min_seq_id: float = 0.3) -> str:
    """Run MMseqs2 clustering on a FASTA and return path to TSV mapping representative->member."""
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    inp_db = os.path.join(out_dir, 'inputDB')
    clu_db = os.path.join(out_dir, 'clusters')
    tsv_path = os.path.join(out_dir, 'clusters.tsv')
    run(['mmseqs', 'createdb', fasta_path, inp_db])
    run(['mmseqs', 'cluster', inp_db, clu_db, tmp_dir, '--min-seq-id', str(min_seq_id), '-c', '0.8', '--cov-mode', '1'])
    run(['mmseqs', 'createtsv', inp_db, inp_db, clu_db, tsv_path])
    return tsv_path


def parse_clusters(tsv_path: str) -> Dict[str, str]:
    """Parse MMseqs2 clusters.tsv into a mapping member->representative."""
    mapping: Dict[str, str] = {}
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep, mem = line.split('\t')[:2]
            mapping[mem] = rep
    return mapping

