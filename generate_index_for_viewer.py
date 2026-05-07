import os
import json


def find_aiscientist_pdf(paper_path):
    folder_name = os.path.basename(os.path.normpath(paper_path)).lower()

    try:
        pdf_files = sorted(
            filename
            for filename in os.listdir(paper_path)
            if filename.lower().endswith('.pdf')
        )
    except OSError:
        return '', ''

    for filename in pdf_files:
        stem = os.path.splitext(filename)[0].lower()
        if stem and stem in folder_name:
            candidate_path = os.path.join(paper_path, filename)
            return './' + candidate_path, 'pdf'

    return '', ''


def find_paper_file(paper_path, base_dir=''):
    if base_dir == 'aiscientist_papers':
        ai_scientist_pdf, paper_type = find_aiscientist_pdf(paper_path)
        if ai_scientist_pdf:
            return ai_scientist_pdf, paper_type

    preferred_candidates = [
        'paper.pdf',
        'paper.md',
        os.path.join('results', 'paper.pdf'),
        os.path.join('results', 'paper.md'),
    ]
    preferred_extensions = ('.pdf', '.md', '.txt')

    for candidate in preferred_candidates:
        candidate_path = os.path.join(paper_path, candidate)
        if os.path.exists(candidate_path):
            rel_path = './' + candidate_path
            ext = os.path.splitext(candidate_path)[1].lower()
            return rel_path, ('pdf' if ext == '.pdf' else 'md')

    try:
        for filename in sorted(os.listdir(paper_path)):
            lower_name = filename.lower()
            if lower_name.endswith(preferred_extensions):
                rel_path = './' + os.path.join(paper_path, filename)
                ext = os.path.splitext(filename)[1].lower()
                return rel_path, ('pdf' if ext == '.pdf' else 'md')
    except OSError:
        return '', ''

    return '', ''

def generate_index():
    """
    Scans specified directories for Claude and Codex analysis results and 
    generates a grouped JSON index for the FabScore viewer.
    """
    tasks = []
    
    # Target directories for active analysis
    search_dirs = [
        'agents4sci_acc', 
        'agents4sci_rej', 
        'aiscientist_papers',
        'mlrbench_papers',
        'fars_papers'
    ]

    print("Scanning task directories for new results...")
    
    for base in search_dirs:
        if not os.path.exists(base):
            continue
            
        for paper in os.listdir(base):
            paper_path = os.path.join(base, paper)
            if not os.path.isdir(paper_path):
                continue

            paper_file, paper_type = find_paper_file(paper_path, base)

            # Check for Claude Audit results (fs_summary.json)
            claude_json = os.path.join(paper_path, 'fabscore_claude', 'fs_summary.json')
            if os.path.exists(claude_json):
                tasks.append({
                    'label': f"{base}: {paper}",
                    'paper': paper_file,
                    'paper_type': paper_type,
                    'pdf': paper_file if paper_type == 'pdf' else '',
                    'json': "./" + claude_json,
                    'type': 'Claude'
                })

            # Check for Codex Audit results (fs_summary.json)
            codex_json = os.path.join(paper_path, 'fabscore_codex', 'fs_summary.json')
            if os.path.exists(codex_json):
                tasks.append({
                    'label': f"{base}: {paper}",
                    'paper': paper_file,
                    'paper_type': paper_type,
                    'pdf': paper_file if paper_type == 'pdf' else '',
                    'json': "./" + codex_json,
                    'type': 'Codex'
                })

    # Sort tasks alphabetically by label
    tasks.sort(key=lambda x: x['label'])

    # Group tasks by model type for the UI selectors
    index_data = {
        'Claude': [t for t in tasks if t['type'] == 'Claude'],
        'Codex': [t for t in tasks if t['type'] == 'Codex']
    }

    output_file = 'paper_index.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4, ensure_ascii=False)
    
    print(f"Index generation complete: {output_file}")
    print(f" - Claude results found: {len(index_data['Claude'])}")
    print(f" - Codex results found: {len(index_data['Codex'])}")

if __name__ == "__main__":
    generate_index()
