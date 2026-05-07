# Validation Scripts

This folder contains Phase 5 validation scripts for the pharmaceutical forecasting system.

## Scripts

### `phase5_real_validation.py`
**Primary validation script** for historical drug performance testing.
- Tests multi-agent system against real drug launches (2015-2020)
- Measures MAPE performance vs consultant baseline (±40% accuracy)
- Current target: ±25% MAPE to achieve consultant-level performance
- Outputs: `results/phase5_real_validation.json`

**Usage:**
```bash
python validation/phase5_real_validation.py
```

### `phase5_data_audit.py`
Data quality auditing and profiling for validation datasets.
- Checks SEC revenue data quality and completeness
- Validates drug launch information
- Identifies data gaps and inconsistencies

**Usage:**
```bash
python validation/phase5_data_audit.py
```

### `phase5_historical_validation.py`
Extended historical validation with broader drug coverage.
- Comprehensive validation across multiple therapeutic areas
- Historical backtesting framework
- Performance benchmarking against industry baselines

**Usage:**
```bash
python validation/phase5_historical_validation.py
```

## Validation Workflow

1. **Data Audit**: Run `phase5_data_audit.py` to check data quality
2. **Real Validation**: Run `phase5_real_validation.py` for core MAPE testing
3. **Historical Validation**: Run `phase5_historical_validation.py` for extended coverage

## Success Criteria

- **Target MAPE**: ≤25% to beat industry consultant baseline (±40%)
- **Current Status**: 81.3% MAPE (ongoing calibration)
- **Drug Differentiation**: ✅ Achieved (different forecasts per drug)
- **System Integration**: ✅ Multi-agent coordination operational

## Results Location

All validation results are stored in `results/` folder:
- `phase5_real_validation.json` - Core validation metrics
- `phase5_data_audit.json` - Data quality reports
- `usage_log.jsonl` - API usage tracking