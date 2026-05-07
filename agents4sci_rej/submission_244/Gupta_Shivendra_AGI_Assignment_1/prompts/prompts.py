# -*- coding: utf-8 -*-
"""
Prompts used for drafting and submission.
Keep this file anonymized (no personal names or affiliations).
"""

# --- Metadata ---
RUN_DATE = "2025-09-15"  # Update if you re-run major drafting passes
PROJECT_TAG = "entropy_pruning_transformer"

# System prompt used during drafting (anonymized)
SYSTEM_PROMPT = """
You are an assistant helping to draft a reproducible ML paper.
Prioritize determinism, artifact logging, and clear compute/ethics disclosures.
Keep claims aligned with saved results. Ensure anonymity for submissions.
"""

# Task prompts (snippets actually used)
TASK_METHOD = """
Draft a clear, minimal method for entropy-guided token pruning:
encoder → attention-1 → entropy gate (top-k) → attention-2 → pool → classifier.
Include budget control (ρ), FLOPs proxy derivation, and a short calibration note.
Avoid overclaiming; cite core works.
"""

TASK_CHECKLIST = """
Fill AI involvement and paper checklist with accurate Yes/No answers
and 1–2 sentence justifications. Ensure template compliance and anonymity.
"""

TASK_README = """
Write a concise README for a NumPy-only synthetic pipeline:
- one command to run experiments (timestamped output),
- where figures/JSON artifacts are saved,
- how to recompute metrics from saved arrays,
- optional SST-2 notebook requirements and caveats.
"""

# ---------------------------------------------------------------------
# Master workflow prompt used during the project (verbatim for provenance)
# ---------------------------------------------------------------------
RESEARCH_WORKFLOW_PROMPT = r"""
---
## STEP 1: RESEARCH IDEA AND CONCEPT DEVELOPMENT
---
## STEP 1: RESEARCH IDEA AND CONCEPT DEVELOPMENT
**Prompt:**
```
Generate a research idea. Focus on:
1. **Research Domain Exploration:**
- Identify the specific AI problem and its current limitations
- Generate unique insights about the problem space
- Understand the practical applications and real-world impact
- Find any specific technologies or methods
2. **Technical Innovation Focus:**
- Identify the specific technical gap to address
- Find concrete technical approaches (preprocessing methods, algorithms)
- Understand how the approach differs from existing solutions
- Find any specific challenges
3. **Implementation Feasibility:**
- Determine appropriate datasets (real or synthetic)
- Identify computational requirements and available resources
- Extract specific evaluation metrics and success criteria
- Find any implementation constraints or preferences
4. **Research Impact Assessment:**
- Identify target applications and user scenarios
- Understand how this advances the field
- Extract potential follow-up research directions
- Find any security or privacy considerations
DELIVERABLE: Generate a comprehensive research outline with:
- Clear problem statement
- Technical approach incorporating specific insights
- Concrete contributions that address identified gaps
- Implementation plan based on constraints
- Evaluation strategy aligned with success criteria
```
## STEP 2: JSON FILE AND CATEGORY GENERATION
**Prompt:**
```
Generate a comprehensive JSON research metadata file that captures the essence of the research:
REQUIRED JSON STRUCTURE:
{
"authors": ["Full Name"],
"instance_id": "descriptive_research_id_yyyy",
"year": 2025,
"url": "",
"abstract": "150-200 word abstract that: 1) States the problem clearly, 2) Describes the novel approach, 3) Highlights key technical contributions, 4) Reports main results with specific numbers, 5) Emphasizes unique advantages",
"venue": "1st Open Conference of AI Agents for Science",
"source_papers": [
{
"reference": "Actual paper title or descriptive placeholder",
"rank": 1,
"type": ["category"], // Options: methodological foundation, signal processing, security analysis, survey, implementation, comparison baseline
"justification": "Specific relevance to the proposed research",
"usage": "Concrete way this will be incorporated"
}
// Include 7-10 key references covering all aspects
],
"task1": "Detailed technical implementation with exact specifications:
1. Data collection/generation parameters
2. Preprocessing pipeline with specific algorithms and parameters
3. Feature extraction methods with mathematical details
4. Model architecture with layer specifications
5. Training protocol with hyperparameters
6. Evaluation metrics and protocols
7. Expected performance targets",
"task2": "Research objectives and expected outcomes with specific targets"
}
CATEGORY FILE (metaprompt.py):
Create a Python file with research framework including:
- TASK: Clear task description from the researcher's perspective
- DATASETS: Specific dataset specifications or generation parameters
- BASELINES: Relevant comparison methods (both traditional and modern)
- EVALUATION: Comprehensive metrics including domain-specific ones
- COMPARISON_TEMPLATE: Table format for results presentation
- ABLATIONS: List of components to analyze
- IMPLEMENTATION: Detailed specifications for reproducibility
DELIVERABLE: Complete JSON file and metaprompt.py with all research specifications.
```
---
## STEP 3: MATHEMATICAL FORMULATION DEVELOPMENT
**Prompt:**
```
Develop comprehensive mathematical foundations as a complete LaTeX document:
MATHEMATICAL COMPONENTS TO DEVELOP:
1. **Signal/Data Representation:**
- Define input data format and dimensions
- Specify any domain-specific representations
- Include preprocessing transformations
2. **Core Algorithm Formulation:**
- Main processing pipeline with clear variable definitions
- Step-by-step mathematical operations
- Include any domain-specific operations
3. **Deep Learning Architecture:**
- Input/output specifications with dimensions
- Layer-wise transformations with activation functions
- Special components (attention, multi-scale processing)
- Loss function design with all terms
4. **Optimization and Training:**
- Total loss function with weighting parameters
- Optimization algorithm and learning rate schedules
- Regularization terms and constraints
- Convergence criteria
5. **Evaluation Framework:**
- Performance metrics with mathematical definitions
- Statistical significance tests
- Domain-specific evaluation criteria
6. **Theoretical Analysis (if applicable):**
- Convergence guarantees
- Computational complexity
- Security/robustness properties
DELIVERABLE: Complete mathematical_formulation.tex with:
- Proper LaTeX formatting and packages
- All equations numbered and labeled
- Clear notation definitions
- Implementation-ready specifications
- Compilation-ready document
```
---
## STEP 4: EXPERIMENTAL IMPLEMENTATION
**Prompt:**
```
Create a complete, runnable Python implementation with realistic simulations:
IMPLEMENTATION REQUIREMENTS:
1. **Dataset Module:**
```python
class ResearchDataset(Dataset):
def __init__(self, data_path, transform=None, mode='train'):
# For demonstration, generate synthetic data that reflects
# the characteristics described in the research
# Include realistic noise and variations
def generate_synthetic_data(self):
# Create data with subject/class-specific patterns
# Add realistic artifacts and variations
```
2. **Preprocessing Pipeline:**
```python
class Preprocessor:
# Implement all preprocessing steps mentioned in research
# Include specific parameters from the interview
# Make it modular and configurable
```
3. **Model Architecture:**
```python
class ProposedModel(nn.Module):
# Implement the exact architecture described
# Include all special components (attention, multi-scale, etc.)
# Make dimensions and hyperparameters configurable
```
4. **Training Framework:**
```python
class Trainer:
# Complete training loop with logging
# Implement custom losses (triplet, center, etc.)
# Include validation and checkpointing
# Add early stopping and learning rate scheduling
```
5. **Evaluation Framework:**
```python
class Evaluator:
# Implement all evaluation metrics
# Include domain-specific evaluations
# Generate comparison tables and statistics
```
6. **Main Experiment Runner:**
```python
def run_comprehensive_experiments():
# Orchestrate full experimental pipeline
# Include ablation studies
# Save all results and checkpoints
# Generate reproducible results
```
IMPORTANT:
- Include realistic synthetic data generation
- Add progress printing and logging
- Ensure the code actually runs without external dependencies
- Include docstrings and comments
DELIVERABLE: Complete Python implementation that can be executed immediately.
```
---
## STEP 5: EXPERIMENT EXECUTION
**Prompt:**
```
Create an experiment execution script that generates realistic results. Use python3 command for running Python code.
EXECUTION REQUIREMENTS:
1. **Experiment Runner Script:**
```python
class ExperimentRunner:
def __init__(self):
self.results = {}
self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
self.output_dir = f"results_{self.timestamp}"
os.makedirs(self.output_dir, exist_ok=True)
```
2. **Baseline Experiments:**
- Generate realistic baseline results based on literature
- Include traditional methods and recent deep learning approaches
- Ensure results reflect expected performance gaps
3. **Main Method Results:**
- Generate main results that show clear improvements
- Include training dynamics (loss curves, accuracy progression)
- Create realistic performance metrics
4. **Ablation Studies:**
- Show contribution of each component
- Generate results that validate design choices
- Include failure modes and limitations
5. **Domain-Specific Analysis:**
- Security analysis (for biometric systems)
- Robustness analysis (adversarial, temporal, etc.)
- Cross-dataset or cross-session performance
6. **Results Storage:**
- Save all results in JSON format
- Generate summary statistics
- Create comparison tables
EXECUTION NOTES:
- Print clear progress messages
- Save results immediately after generation
- Include timestamps and versioning
- Generate results that tell a coherent story
DELIVERABLE: Executable script that generates complete experimental results.
```
---
## STEP 6: RESULTS GENERATION AND VISUALIZATION
**Prompt:**
```
Generate publication-quality figures using matplotlib/seaborn:
VISUALIZATION REQUIREMENTS:
1. **Setup:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-paper') # Use versioned style
sns.set_palette("husl")
```
2. **Required Figures:**
- Training/validation curves (loss and accuracy)
- ROC curves with AUC values
- Performance comparison bar charts
- Ablation study visualizations (multi-panel)
- Domain-specific visualizations
- Cross-session/temporal performance
3. **Figure Guidelines:**
- High resolution (300 DPI)
- Clear labels and legends
- Consistent color scheme
- Both PDF and PNG formats
- Proper font sizes for paper inclusion
4. **Special Visualizations:**
- Confusion matrices (if applicable)
- Feature importance or attention maps
- Statistical significance indicators
- Error bars and confidence intervals
5. **Tables as Figures:**
- Create visual tables for key comparisons
- Highlight best results
- Include statistical significance markers
DELIVERABLE: Complete set of publication-ready figures saved in results directory.
```
---
## STEP 7: RESULTS ANALYSIS
**Prompt:**
```
Conduct comprehensive analysis of experimental results:
ANALYSIS FRAMEWORK:
1. **Performance Analysis:**
- Interpret quantitative metrics in context
- Compare with baselines and state-of-the-art
- Identify strengths and weaknesses
- Explain unexpected results
2. **Technical Analysis:**
- Why does the method work?
- Which components contribute most?
- What are the failure modes?
- How does it scale?
3. **Ablation Insights:**
- Validate each design decision
- Identify critical vs. optional components
- Suggest potential simplifications
4. **Domain-Specific Analysis:**
- Security implications (for biometric systems)
- Robustness characteristics
- Practical deployment considerations
- Computational requirements
5. **Statistical Analysis:**
- Significance of improvements
- Confidence intervals
- Effect sizes
- Multiple comparison corrections
DELIVERABLE: Comprehensive analysis document (results_analysis.md) with:
- Executive summary
- Detailed findings
- Implications for practice
- Future research directions
- Honest discussion of limitations
```
---
## STEP 8: PAPER WRITING
**Prompt:**
```
Write a complete paper in LaTeX format following conference guidelines
https://agents4science.stanford.edu/assets/Agents4Science_Template.zip:
PAPER REQUIREMENTS:
1. **LaTeX Setup:**
- Use standard article class
- Include all necessary packages
- Set appropriate margins and formatting
- Ensure compilation compatibility
2. **Paper Structure:**
- Title: Descriptive and impactful
- Abstract: 150-200 words with key contributions and results
- Introduction: Problem motivation, contributions, paper organization
- Related Work: Comprehensive but concise literature review
- Method: Detailed technical description with subsections
- Experiments: Setup, results, ablations, analysis
- Conclusion: Summary and future work
- References: Properly formatted citations
3. **Figure Integration:**
- Include all generated figures with \includegraphics
- Add informative captions
- Reference all figures in text
- Ensure proper placement
4. **Writing Guidelines:**
- Clear, concise technical writing
- Define all notation before use
- Include enough detail for reproduction
- Balance technical depth with readability
5. **Length and Formatting:**
- Target 8 pages plus references (or workshop limit)
- Two-column format
- Proper section numbering
- Consistent formatting throughout
IMPORTANT: Create a full version (with the conference style if available) and a that compiles in LaTeX.
DELIVERABLE: Complete LaTeX paper ready for compilation with pdflatex.
```
---
## STEP 9: REVIEW GENERATION
**Prompt:**
```
Generate a realistic academic review that helps improve the paper:
REVIEW STRUCTURE:
1. **Summary** (2-3 sentences)
2. **Scores:**
- Technical Quality (1-10)
- Clarity and Presentation (1-10)
- Significance and Impact (1-10)
- Experimental Evaluation (1-10)
3. **Strengths** (5-7 bullet points)
- Technical contributions
- Experimental rigor
- Practical applications
- Writing quality
4. **Weaknesses** (5-7 bullet points)
- Constructive criticism
- Missing experiments
- Unclear explanations
- Limited evaluations
5. **Detailed Comments:**
- Technical soundness
- Experimental completeness
- Comparison with state-of-the-art
- Reproducibility concerns
6. **Questions for Authors**
7. **Recommendation:** Accept/Weak Accept/Weak Reject/Reject
8. **Confidence Level:** 1-5
DELIVERABLE: Detailed review that identifies real improvement opportunities.
```
---
## STEP 10: FINAL SUBMISSION PREPARATION
**Prompt:**
```
Prepare the complete submission package:
FINAL CHECKLIST:
1. **Core Files:**
- Main paper (both .tex and .pdf)
- All figures in high resolution
- Bibliography/references
- Mathematical formulations
2. **Supplementary Materials:**
- Complete code implementation
- Detailed experimental results
- Additional analyses
- README with instructions
3. **Compilation:**
- Run pdflatex twice to resolve references
- Verify all figures are included
- Check total page count
- Ensure PDF/A compliance
4. **Final Verification:**
- All claims supported by experiments
- Figures and tables properly referenced
- No compilation warnings
- Meets submission guidelines
IMPORTANT: Always compile the LaTeX into a final PDF and verify it includes all figures.
DELIVERABLE: Publication-ready PDF with all supplementary materials organized.
```
---
## EXECUTION WORKFLOW
**Enhanced Execution Guidelines:**
1. **Initialize:** Read interview transcript carefully
2. **Plan:** Create todo list and track progress throughout
3. **Implement:** Follow steps sequentially, building on previous outputs
4. **Verify:** Test all code and compile all LaTeX documents
5. **Iterate:** Address any issues before proceeding
6. **Finalize:** Ensure PDF includes all figures and supplementary materials
**Key Improvements from Experience:**
- Always use python3 for Python execution
- Generate synthetic data for demonstrations
- Include figures in final paper compilation
- Create both simplified and full LaTeX versions
- Save results immediately with timestamps
- Test all code before moving to next step
- Track progress with todo lists
"""
