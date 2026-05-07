# Ablation: Contribution of href-Rewriting in XBRL Package Reconstruction

## Experiment Overview

This ablation measures how critical the href-rewriting step is in the XBRL package reconstruction pipeline. The reconstruction pipeline (in `data/xbrl_reconstructor.py`) performs several steps:

1. Section splitting (extracting schema, linkbase, instance XML from query text)
2. File writing (saving each section as a separate XML file)
3. XML sanitization (fixing double-encoded entities, repairing truncated XML)
4. **href rewriting** (renaming schemaRef to local `schema.xsd`, injecting local linkbaseRef entries)

This ablation disables step 4 only, keeping all other steps intact. The reconstructed XBRL files retain their original `xlink:href` values from the query text (which point to company-specific filenames like `company-20230101.xsd` that do not exist locally).

## Setup

- **Dataset**: TheFinAI/FinMR, 332 instances (test split)
- **Baseline**: Arelle symbolic baseline with href rewriting enabled (195/332 executable, ACC=42.17%)
- **Ablation**: Same pipeline with `skip_href_rewrite=True`
- **Arelle mode**: Offline (`internetConnectivity="offline"`) -- cannot fetch remote URLs

## Key Results

| Variant | Executability Coverage | ACC on Executable Subset | Dominant Failure Cause |
|---------|----------------------|--------------------------|----------------------|
| With href rewrite (baseline) | 58.73% (195/332) | 71.79% | N/A |
| Without href rewrite (ablation) | **0.00% (0/332)** | N/A (no executable instances) | missing_dts_artifact |

## Key Observations

1. **Total executability loss**: Disabling href rewriting reduces executability from 58.73% to 0.00%. Not a single instance remains executable. This is the strongest possible ablation signal -- href rewriting is indispensable.

2. **Failure mechanism**: All 332 instances fail with `missing_dts_artifact`. The root cause is that the instance document's `schemaRef` attribute points to the original company-specific schema filename (e.g., `abc-20230630.xsd`), but the reconstructed schema is always saved as `schema.xsd`. Without rewriting `schemaRef` to `schema.xsd`, Arelle cannot find the schema, and without the schema, it cannot load any DTS (Discoverable Taxonomy Set) components.

3. **Two rewriting operations are both critical**:
   - **Schema name rewriting**: `orig_schema -> schema.xsd` in instance.xml and linkbase files
   - **Linkbase ref injection**: Adding `linkbaseRef` entries in the schema so Arelle discovers the local linkbase files (presentation, calculation, definition, label)

4. **FinMR queries are NOT self-contained**: The query text embeds XML document content but uses the original filing-specific filenames in cross-references. Without the reconstruction pipeline's href normalization, these references are unresolvable in any offline execution context.

5. **Implication for the research proposal**: The href-rewriting step is a prerequisite for any symbolic execution approach on FinMR. Any future method that loads FinMR XBRL with a standards-compliant parser must implement equivalent reference resolution.
