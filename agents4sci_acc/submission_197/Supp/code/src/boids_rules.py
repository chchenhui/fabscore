import itertools
import statistics
from typing import List, Dict, Any, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# A simplified representation of a tool's metadata, used for Boids calculations.
ToolMetadata = Dict[str, Any]

def _read_tool_code(tool_meta: Dict, base_dir: str) -> str:
    """Safely reads the code of a tool from its file."""
    try:
        file_path = os.path.join(base_dir, tool_meta['file'])
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                # Return the first 20 lines for brevity
                return "".join(f.readlines()[:20])
        return "# Code not available"
    except Exception:
        return "# Error reading code"

def prepare_alignment_prompt(neighbor_tools_meta: List[Dict], current_round: int, tools_base_dir: str, rounds_window: int = 3) -> str:
    """
    Generates a prompt snippet for the ALIGNMENT rule, focusing on learning from successful neighbors.
    Now includes code snippets.
    """
    # 1. Filter for tools created in the recent window.
    recent_tools = [
        t for t in neighbor_tools_meta 
        if "created_in_round" in t and t["created_in_round"] >= (current_round - rounds_window)
    ]
    
    if not recent_tools:
        return ""

    # 2. Find "success" exemplars.
    # Complexity Leader
    complexity_leader = max(recent_tools, key=lambda t: t.get("complexity", {}).get("tci_score", 0), default=None)
    
    # Quality Leader
    quality_leader = max(
        (t for t in recent_tools if t.get("test_passed") is True),
        key=lambda t: t.get("complexity", {}).get("tci_score", 0), # Tie-break with complexity
        default=None
    )

    # Adoption Leader
    adoption_leader = max(
        (t for t in recent_tools if t.get("adoption_count", 0) > 0),
        key=lambda t: t.get("adoption_count", 0),
        default=None
    )

    # 3. Build the prompt string.
    prompt_lines = ["[🎯 ALIGNMENT: Learn from Successful Neighbors]",
                    "Your neighbors have recently built some remarkable tools. Learn from their strategies:"]

    if complexity_leader:
        tci_score = complexity_leader.get("complexity", {}).get("tci_score", 0)
        code_snippet = _read_tool_code(complexity_leader, tools_base_dir)
        prompt_lines.append(f"\nComplexity Leader (TCI Score: {tci_score:.2f}):")
        prompt_lines.append(f"- Tool: '{complexity_leader['name']}' by {complexity_leader['created_by']}")
        prompt_lines.append(f"- Strategy: This tool achieves high complexity. Analyze its code for composition and depth.")
        prompt_lines.append(f"  - Code:\n```python\n{code_snippet}\n```")

    if quality_leader:
        code_snippet = _read_tool_code(quality_leader, tools_base_dir)
        prompt_lines.append(f"\nQuality Leader (Tests: 100% Passed):")
        prompt_lines.append(f"- Tool: '{quality_leader['name']}' by {quality_leader['created_by']}")
        prompt_lines.append(f"- Strategy: This tool is highly reliable. Analyze its code for robust error handling and clear functionality.")
        prompt_lines.append(f"  - Code:\n```python\n{code_snippet}\n```")

    if adoption_leader:
        adoption_count = adoption_leader.get("adoption_count", 0)
        code_snippet = _read_tool_code(adoption_leader, tools_base_dir)
        prompt_lines.append(f"\nAdoption Leader (Used by {adoption_count} other tools):")
        prompt_lines.append(f"- Tool: '{adoption_leader['name']}' by {adoption_leader['created_by']}")
        prompt_lines.append(f"- Strategy: This tool is the most widely used by others. Analyze its simple API and focused functionality to understand why it's so effective.")
        prompt_lines.append(f"  - Code:\n```python\n{code_snippet}\n```")

    if not complexity_leader and not quality_leader and not adoption_leader:
        return "" # No exemplars to learn from

    prompt_lines.append("\nYOUR GOAL: Do not just copy their code. Instead, adopt their successful DESIGN PRINCIPLES like modularity, composition, and robustness in your own unique tool.")
    return "\n".join(prompt_lines)

def prepare_separation_prompt(neighbor_tools_meta: List[Dict], tools_base_dir: str, 
                            similarity_threshold: float = 0.3, top_n: int = 2) -> str:
    """
    Generates a prompt snippet for the SEPARATION rule, warning about functionally similar tools.
    Now includes code snippets and is dynamic.
    """
    if len(neighbor_tools_meta) < 2:
        return ""

    # Combine name and description for semantic analysis
    documents = [f"{tool['name']} {tool['description']}" for tool in neighbor_tools_meta]
    
    # Calculate TF-IDF and cosine similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        # TfidfVectorizer can fail if all documents are empty or stop words
        return ""

    # Find the top N most similar pairs of tools to any other tool
    similarities = []
    for i in range(len(neighbor_tools_meta)):
        for j in range(i + 1, len(neighbor_tools_meta)):
            similarities.append(((i, j), cosine_sim_matrix[i, j]))
    
    # Sort by similarity and get unique tools from top pairs
    sorted_sim = sorted(similarities, key=lambda item: item[1], reverse=True)
    
    top_tools_indices = []
    seen_indices = set()
    for (idx1, idx2), sim in sorted_sim:
        if sim < similarity_threshold: # Use parameter instead of hardcoded value
            break
        if idx1 not in seen_indices:
            top_tools_indices.append((idx1, sim))
            seen_indices.add(idx1)
        if idx2 not in seen_indices:
            top_tools_indices.append((idx2, sim))
            seen_indices.add(idx2)
        if len(top_tools_indices) >= top_n:
            break
            
    if not top_tools_indices:
        return ""

    prompt_lines = ["[🚧 SEPARATION: Avoid Redundancy]",
                    "Your neighbors' tools show some functional clusters. Be aware of these to ensure your contribution is unique:"]

    for i, (tool_idx, sim) in enumerate(top_tools_indices[:top_n]):
        tool = neighbor_tools_meta[tool_idx]
        code_snippet = _read_tool_code(tool, tools_base_dir)
        prompt_lines.append(f"\nMost Similar Neighbor Tool #{i+1} ('{tool['name']}' with ~{sim:.0%} similarity to others):")
        prompt_lines.append(f"- Description: {tool['description']}")
        prompt_lines.append(f"- Code:\n```python\n{code_snippet}\n```")

    prompt_lines.append("\nYOUR GOAL: Ensure your tool offers a DISTINCT function. Do not rebuild a tool that performs these core tasks. Find a new, complementary niche.")
    
    return "\n".join(prompt_lines)

def prepare_cohesion_prompt(global_summary: str) -> str:
    """
    Generates a prompt snippet for the COHESION rule, using a global summary.
    """
    if not global_summary:
        return ""
    
    prompt_lines = ["[🌍 COHESION: Align with the Global Trend]",
                    "An observer has summarized the entire society's activity from the last round. This is the current 'center of gravity' for our ecosystem:",
                    f'\n"{global_summary}"',
                    "\nYOUR GOAL: Contribute to this emerging trend. How can your tool serve this broader goal?"]
    
    return "\n".join(prompt_lines)

def prepare_self_reflection_prompt(reflection_history: List[Dict]) -> str:
    """
    Generates a prompt snippet that shows the agent its last reflection to prevent loops.
    """
    if not reflection_history:
        return ""

    last_reflection = reflection_history[-1]
    last_reflection_text = last_reflection.get("reflection", "No reflection text found.")

    prompt_lines = ["[MEMORY: Your Previous Reflection]",
                    "Review your most recent thought to ensure you are making forward progress.",
                    f"\nYour Last Idea: \"{last_reflection_text[:300]}...\"", # Truncate for brevity
                    "\nYOUR GOAL: Build upon this previous idea, refine it, or explicitly choose a new direction. **Avoid proposing the exact same tool or strategy again.**"]

    return "\n".join(prompt_lines) 