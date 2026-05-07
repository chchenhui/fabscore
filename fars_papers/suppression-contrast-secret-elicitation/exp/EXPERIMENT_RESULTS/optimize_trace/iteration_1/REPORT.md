# Optimization Iteration 1: Cross-Model Generic Token Filter

## Experiment Overview

Explored multiple approaches to improve SCT beyond the step-0 optimized ctrl_sct_a0 (TR@5=4.67%):
1. **Differential SCT** (finetuned vs base model contrast) - FAILED
2. **Multi-position aggregation** variants - FAILED
3. **Cross-model generic token filter** - Best improvement found

The cross-model generic filter removes tokens that appear frequently in top-K rankings across ALL other finetuned models. These are "generic game tokens" (e.g., "guess", "hints", "Sorry") that are suppressed by all models regardless of the secret word, not secret-specific signal.

## Setup

- **Models**: Gemma-2-9B-IT finetuned on Taboo (gold, moon, flag)
- **Base model**: google/gemma-2-9b-it (for differential approach)
- **Auditor**: gemini-3-flash via LEMMA_MAAS API (gemma-3-4b-it unavailable)
- **Scoring positions**: Control tokens (positions 0,1 of assistant turn)
- **Mid layer**: 32, **Final layer**: 41
- **Top-K stored**: 200 per position per layer
- **Cross-model filter params**: top_k_scan=20, threshold_pct=1.0%

## Key Results

All results using gemini-3-flash auditor for fair comparison.

| Method | gold TR@5 | moon TR@5 | flag TR@5 | Mean TR@5 | gold Aud | moon Aud | flag Aud | Mean Aud |
|--------|-----------|-----------|-----------|-----------|----------|----------|----------|----------|
| LL baseline (a=0.1) | 2.0% | 7.0% | 4.0% | 4.33% | 1.2% | 5.0% | 3.2% | 3.13% |
| Ctrl SCT a0 (step0) | 2.0% | 8.0% | 4.0% | 4.67% | 0.7% | 3.8% | 2.1% | 2.20% |
| Filtered SCT | 2.0% | 10.0% | 4.0% | 5.33% | 1.0% | 5.0% | 2.1% | 2.70% |
| **Filtered LL (best)** | **2.0%** | **10.0%** | **4.0%** | **5.33%** | **1.0%** | **5.8%** | **2.4%** | **3.07%** |

### Success Criteria Check (vs LL baseline)

- TR@5 relative improvement: +23.1% (need >=25%) -- NOT MET
- TR@5 absolute improvement: +1.00pp (need >=1.5pp) -- NOT MET
- Auditor accuracy change: -0.06pp -- NOT IMPROVED

## Key Observations

1. **Cross-model filter helps moon significantly**: Moon TR@5 improved from 7.0% to 10.0% by removing generic game tokens that were displacing the secret.

2. **Gold and flag unchanged**: Gold (2.0%) and flag (4.0%) TR@5 are unchanged. Gold is limited by the top-200 ceiling (only 2% of examples have gold in top-200 at control positions). Flag is stuck because model-specific noise tokens ("hello", "Hello", "Greetings") aren't filtered by cross-model approach.

3. **Differential SCT failed**: Both diff_SCT and mid_diff scored full-vocabulary but generic tokens still dominated. The assumption that base model encodes similar suppression patterns was incorrect -- the base model produces very different activation patterns on the Taboo-specific prompts.

4. **LL with filter >= SCT with filter**: After removing generic tokens, the simpler logit lens method matches or exceeds SCT on both TR@5 and auditor accuracy. The suppression contrast signal provides no additional benefit once generic noise is removed.

5. **Fundamental limitation**: The top-200 token ceiling at control positions constrains all within-extraction approaches. The theoretical maximum recovery rate is limited by how often the secret appears in the pre-extracted top-200 tokens.

## Approaches Explored

### Differential SCT (FAILED)
- `diff_SCT(v) = SCT_ft(v) - SCT_base(v)` at full vocabulary
- Also tested: `mid_diff(v) = log_p_mid_ft(v) - log_p_mid_base(v)`
- Result: All mid_diff scores negative, gold not in top-500. Generic tokens still dominate.

### Multi-position Aggregation (FAILED)
- Tested max, vote, sum across all positions; content position bonus
- All worse than control-only scoring

### Cross-model Generic Token Filter (BEST)
- For target model X, compute SCT rankings for other two models
- Tokens in top-20 for >=1% of examples in ALL other models are "generic"
- Remove generic tokens from X's rankings before evaluation

### LL-Filtered (BEST OVERALL)
- Logit lens at control positions (alpha=0) + cross-model generic filter
- Same TR@5 as filtered SCT, better auditor accuracy

## Files

- `taboo_logit_lens_baseline_auditor_gemini.json` - LL baseline auditor results
- `taboo_ctrl_sct_a0_auditor_gemini.json` - Step-0 SCT auditor results
- `taboo_filtered_sct_auditor.json` - Filtered SCT auditor results
- `taboo_filtered_ll_auditor.json` - Filtered LL auditor results (best)
- Scored outputs: `sct/outputs/taboo_{gold,moon,flag}_filtered_{sct,ll}_scored.json`
- Filter scorer: `sct/extraction/sct_scorer_filtered.py`, `sct/extraction/ll_scorer_filtered.py`
- Differential extraction: `sct/extraction/extract_differential_sct.py`
