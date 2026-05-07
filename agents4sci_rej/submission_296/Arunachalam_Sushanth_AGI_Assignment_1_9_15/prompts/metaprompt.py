"""
Metaprompt configuration for Intelligent Document Processing (IDP) Graduate Admissions System
Contains task definitions, dataset specifications, baselines, evaluation metrics, and implementation details
"""

TASK = """
Intelligent Document Processing for Graduate Admissions Pre-Screening

Goal: Automate academic readiness assessment from application documents (transcripts, resumes, statements of purpose)
through OCR-based parsing, structured information extraction, calibrated decision-making, and human-in-the-loop escalation.

Core Pipeline:
1. OCR/Parse: Extract text and structure from PDF documents using configurable backends (pdfminer, simulated, optional pytesseract)
2. Transcript Analysis: Parse course rows, credits, grades → compute GPA with evidence spans
3. Resume Processing: Extract education, experience, skills via lightweight NER  
4. Statement Evaluation: Multi-label rubric classification with cited-span summarization
5. Feature Fusion: Combine academic, experiential, and narrative features
6. Decision Engine: Apply configurable thresholds with confidence-based abstention
7. Evidence Grounding: Link decisions to specific document spans for transparency
8. UI Dashboard: Interactive interface for review, escalation, and threshold management

Output: Structured JSON with decision, confidence, evidence spans, and human escalation triggers
"""

DATASETS = """
Synthetic Dataset Generation (Privacy-Safe, No PII):

1. Transcripts (N=1000):
   - Universities: 10 institutions with distinct formatting templates
   - Templates: 5 layout variations per university (table vs. list vs. mixed)
   - Variations: 20 GPA/credit combinations per template
   - GPA Distribution: Normal(μ=3.2, σ=0.6), range [2.0, 4.0]
   - Credit Distribution: Uniform[60, 180] total credits
   - Courses: 40+ course pool (CS/MATH/PHYS/ENG) with realistic grade patterns
   - Quality Levels: Clean OCR, moderate noise, high noise (scanning artifacts)

2. Resumes (N=500):
   - Experience Levels: 0-8 years distributed exponentially (λ=0.3)
   - Education: Bachelor's (80%), Master's (15%), PhD (5%)
   - Skills: Technical skills from controlled vocabulary (40+ programming languages, tools, frameworks)
   - Format Variations: 3 standard templates with different section ordering
   - Industries: Tech (60%), Academia (25%), Other (15%)

3. Statements of Purpose (N=300):
   - Length: 300-800 words, Normal(μ=500, σ=120)
   - Rubric Dimensions: Research Interest (0-5), Experience Relevance (0-5), Writing Quality (0-5), 
     Goal Clarity (0-5), Fit Assessment (0-5)
   - Content Types: Research-focused (50%), Industry-focused (30%), Career-change (20%)
   - Quality Distribution: Bimodal with peaks at scores 2-3 and 4-5

Ground Truth Labels:
- Academic Decisions: ACCEPT_ACADEMIC (30%), REVIEW (40%), REJECT_ACADEMIC (25%), ABSTAIN (5%)
- Confidence Scores: Beta(α=2, β=2) scaled to [0.4, 0.95]
- Evidence Spans: Character-level annotations for GPA sources, skills, rubric criteria matches
"""

BASELINES = """
Comparison Methods:

1. GPA-Only Baseline:
   - Simple threshold: GPA ≥ 3.0 → ACCEPT, GPA < 2.5 → REJECT, else REVIEW
   - No abstention mechanism
   - No evidence grounding
   - Manual GPA entry (no OCR)

2. Random Baseline:
   - Random decisions with class priors matching dataset distribution
   - Uniform confidence scores [0.5, 1.0]
   - No evidence or reasoning

3. OCR-Free Oracle:
   - Perfect text extraction (no OCR errors)
   - Ground truth course parsing
   - Tests decision logic independent of OCR quality

4. Feature Ablations:
   - Transcript-Only: No resume or SoP features
   - No-Calibration: Raw model outputs without temperature scaling
   - No-Layout: Text-only processing without spatial/structural cues
   - Single-Threshold: Fixed decision boundary without program-specific rules
"""

EVALUATION = """
Comprehensive Evaluation Metrics:

1. Academic Performance Metrics:
   - GPA Extraction: Mean Absolute Error (MAE), Root Mean Square Error (RMSE)
   - Credit Extraction: Exact match accuracy, ±5 credit tolerance
   - Course Parsing: Row-level F1-score, character-level span overlap

2. Decision Quality Metrics:
   - Academic Classification: Precision, Recall, F1-score per class
   - ROC-AUC and Precision-Recall AUC
   - Confusion Matrix with cost-weighted analysis
   - Rank Correlation: Kendall τ between predicted and ground truth scores

3. Calibration and Reliability:
   - Expected Calibration Error (ECE) with 10 bins
   - Maximum Calibration Error (MCE)
   - Reliability Diagrams (confidence vs. accuracy)
   - Abstention Coverage: % cases with confidence < threshold

4. Information Extraction Quality:
   - NER Performance: Entity-level F1-score (PER/ORG/SKILL/EDU)
   - Summarization Quality: ROUGE-L, BLEU-4 against ground truth bullets
   - Evidence Grounding: Span overlap IoU, citation accuracy

5. Operational Impact:
   - Processing Time: Seconds per document, throughput (docs/hour)
   - Human Agreement: κ-agreement on escalated cases
   - Time Savings: Estimated manual review time reduction
   - Error Analysis: Failure mode categorization and frequency

6. Fairness and Robustness:
   - Demographic Parity: Decision rates across synthetic demographic groups
   - Document Quality Robustness: Performance vs. OCR noise levels
   - Template Generalization: Cross-university performance variance
   - Threshold Sensitivity: Performance stability across parameter ranges

Target Performance:
- GPA MAE < 0.1, Credit Accuracy > 95%
- Academic Classification AUC > 0.85, ECE < 0.1  
- NER F1 > 0.8, ROUGE-L > 0.6
- Processing Time < 30 seconds/application
- Human Escalation Rate < 15% with >90% agreement
"""

COMPARISON_TEMPLATE = """
Performance Comparison Table Schema:

| Method | GPA MAE | Credit Acc | Academic AUC | ECE | NER F1 | ROUGE-L | Time (s) | Escalation % |
|--------|---------|------------|--------------|-----|--------|---------|----------|--------------|
| Random Baseline | {mae:.3f} | {acc:.1f}% | {auc:.3f} | {ece:.3f} | {ner:.3f} | {rouge:.3f} | {time:.1f} | {esc:.1f}% |
| GPA-Only | {mae:.3f} | {acc:.1f}% | {auc:.3f} | {ece:.3f} | - | - | {time:.1f} | {esc:.1f}% |
| OCR-Free Oracle | {mae:.3f} | {acc:.1f}% | {auc:.3f} | {ece:.3f} | {ner:.3f} | {rouge:.3f} | {time:.1f} | {esc:.1f}% |
| Proposed IDP | **{mae:.3f}** | **{acc:.1f}%** | **{auc:.3f}** | **{ece:.3f}** | **{ner:.3f}** | **{rouge:.3f}** | **{time:.1f}** | **{esc:.1f}%** |

Ablation Study Results:

| Feature Set | Academic AUC | ECE | Processing Time | Notes |
|------------|--------------|-----|------------------|-------|
| Full Pipeline | {auc:.3f} | {ece:.3f} | {time:.1f}s | Complete system |
| No Resume/SoP | {auc:.3f} | {ece:.3f} | {time:.1f}s | Transcript-only |
| No Calibration | {auc:.3f} | {ece:.3f} | {time:.1f}s | Raw confidences |
| No Layout Cues | {auc:.3f} | {ece:.3f} | {time:.1f}s | Text-only processing |
| Single Threshold | {auc:.3f} | {ece:.3f} | {time:.1f}s | No program-specific rules |
"""

ABLATIONS = """
Systematic Ablation Studies:

1. Input Channel Ablations:
   - Transcript-Only: Remove resume and SoP features, test academic signal sufficiency
   - No-Resume: Remove experience/skills features, evaluate academic+narrative combination  
   - No-SoP: Remove statement features, test quantitative-only decisions
   - Academic-Only: GPA/credits only, minimal feature set baseline

2. Processing Component Ablations:
   - No-OCR: Use ground truth text, isolate parsing vs. extraction errors
   - No-Layout: Ignore spatial/structural document cues, pure text processing
   - No-NER: Skip entity recognition, use keyword matching only
   - Raw-Text: No preprocessing, tokenization, or normalization

3. Decision Framework Ablations:
   - No-Calibration: Skip temperature scaling, use raw model confidences
   - No-Abstention: Force decisions without confidence-based escalation
   - Single-Threshold: Use global thresholds vs. program-specific rules
   - Hard-Decisions: Remove confidence scoring, binary accept/reject only

4. Feature Engineering Ablations:
   - No-Normalization: Raw GPA/credit values without standardization
   - No-Interaction: Linear features only, no cross-document combinations
   - Minimal-Features: Core academic metrics only (GPA, credits, major)
   - Extended-Features: Add derived signals (grade trends, course difficulty)

Expected Results:
- Full pipeline should outperform all ablations
- Calibration critical for reliable confidence estimation
- Multi-document features improve borderline case handling
- Layout cues essential for accurate transcript parsing
"""

IMPLEMENTATION = """
Reproducibility and Implementation Details:

1. Environment Requirements:
   - Python 3.8+ (Windows compatible)
   - Dependencies: numpy, pandas, scikit-learn, matplotlib, scipy, pillow, pdfminer.six, pyyaml, streamlit
   - Optional: seaborn (plotting), pytesseract (OCR), fastapi/uvicorn (API)
   - Hardware: CPU-only, 8GB RAM recommended, 2GB storage

2. Configuration Management:
   - config/config.yaml: All thresholds, paths, model parameters
   - Environment variables: PYTHONPATH, LOG_LEVEL, RANDOM_SEED
   - Version pinning: requirements.txt with exact versions
   - Docker alternative: Dockerfile for containerized execution

3. Reproducibility Protocol:
   - Fixed random seeds: numpy (42), sklearn models (42), data generation (42)
   - Deterministic synthetic data generation with documented parameters
   - Version control: git tags for experimental snapshots
   - Results archiving: timestamped results/{YYYYmmdd_HHMMSS}/ directories

4. Execution Commands:
   - Full experiment: `python code/run_experiments.py`
   - Watch-folder service: `python code/ingest_service.py --watch --config config/config.yaml`
   - CLI processing: `python code/cli.py ingest --src incoming --backend pdfminer`
   - UI dashboard: `streamlit run ui/app.py --server.port 8501`
   - Test suite: `python -m pytest code/tests/ -v`

5. Data Management:
   - Synthetic data generation: Automated with configurable parameters
   - Privacy compliance: No PII in generated data, anonymized examples only
   - Storage structure: incoming/, processed/, rejected/, archive/, logs/
   - Export formats: JSON, CSV, PDF reports

6. Performance Monitoring:
   - Logging: structured logs to logs/service.log with timestamps
   - Metrics tracking: results/metrics.json with full experimental results
   - Error handling: comprehensive try/catch with informative error messages
   - Health checks: System resource monitoring and processing queue status

7. Deployment Considerations:
   - Local-only execution (no external dependencies)
   - Windows service registration options
   - Batch vs. real-time processing modes  
   - Integration APIs for existing admission systems
   - Backup and recovery procedures for critical processing
"""

# Configuration validation
def validate_config():
    """Validate that all required configuration sections are present"""
    required_sections = ['TASK', 'DATASETS', 'BASELINES', 'EVALUATION', 'COMPARISON_TEMPLATE', 'ABLATIONS', 'IMPLEMENTATION']
    for section in required_sections:
        if section not in globals():
            raise ValueError(f"Missing required configuration section: {section}")
    return True

if __name__ == "__main__":
    validate_config()
    print("✓ Metaprompt configuration validated successfully")
    print(f"Task: {TASK[:100]}...")
    print(f"Dataset size: {DATASETS.count('N=')} dataset specifications")
    print(f"Baselines: {len([line for line in BASELINES.split('\n') if line.strip().endswith(':')])} methods")
    print(f"Evaluation metrics: {EVALUATION.count('- ')} metric categories")