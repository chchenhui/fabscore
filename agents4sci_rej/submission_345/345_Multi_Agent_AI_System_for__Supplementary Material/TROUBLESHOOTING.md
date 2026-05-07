# Troubleshooting Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests to validate installation  
python -m pytest tests/ -v

# 3. Launch Streamlit app
python -m streamlit run src/app.py
```

## Common Issues

### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'matplotlib'` (or other modules)

**Solution**:
```bash
pip install matplotlib seaborn scipy pandas streamlit pytest
```

**Root Cause**: Missing dependencies not installed during initial setup.

### 2. Streamlit App Won't Start

**Problem**: `streamlit: command not found` or import errors

**Solution**:
```bash
# Use python -m prefix
python -m streamlit run src/app.py

# Or install streamlit explicitly
pip install streamlit
```

### 3. Test Failures

**Problem**: Some tests failing after fresh install

**Solution**:
```bash
# Run specific test to see error
python -m pytest tests/test_npv.py -v

# Usually missing scipy
pip install scipy scikit-learn
```

### 4. Configuration Issues

**Problem**: Config file not found or loading errors

**Solution**: 
- Ensure `conf/params.yml` exists
- Check YAML syntax is valid
- Run from project root directory

### 5. Chart Display Issues

**Problem**: Charts not showing in Streamlit app

**Solution**:
```bash
# Install all plotting dependencies
pip install matplotlib seaborn
```

**Note**: Restart Streamlit app after installing new packages.

## Validation Commands

### Quick Health Check
```bash
# Test core models
python src/models/bass.py
python src/econ/npv.py

# Run all tests
python -m pytest tests/ -v

# Validate imports
python -c "from src.models.bass import bass_adopters; print('OK')"
```

### Component Testing
```bash
# Test specific modules
python -m pytest tests/test_bass.py -v        # Bass diffusion
python -m pytest tests/test_npv.py -v         # NPV/Monte Carlo  
python -m pytest tests/test_access_rules.py -v # Pricing simulator
```

## Performance Issues

### Slow Monte Carlo
- Reduce simulation count in app (default: 5000)
- Check available RAM (10k simulations ≈ 100MB)

### App Loading Slowly
- Clear Streamlit cache: Delete `.streamlit` folder
- Restart app completely

## Known Limitations

1. **Export Features**: LaTeX/PowerPoint export not yet implemented
2. **Data Scale**: Only works with small synthetic datasets currently  
3. **Windows Paths**: Some file paths may need adjustment on different systems

## Getting Help

1. **Check all dependencies installed**: `pip list | grep -E "(pandas|numpy|streamlit|matplotlib)"`
2. **Run validation script**: See commands above
3. **Check app is accessible**: Navigate to http://localhost:8501
4. **Review error messages**: Most issues are missing dependencies

## Environment Setup

### Recommended Python Version
- Python 3.10+ (tested on 3.10.11)
- Windows 10/11 or modern Linux/macOS

### Virtual Environment
```bash
# Recommended: Use virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### IDE Configuration
- Ensure IDE uses correct Python interpreter (from .venv)
- Add `src/` to Python path if running individual modules

## Success Indicators

When everything works correctly:
- ✅ All 68 tests pass
- ✅ Streamlit app loads at http://localhost:8501  
- ✅ Parameter sliders update charts in real-time
- ✅ Monte Carlo button generates distribution plots
- ✅ No error messages in browser console

## Contact

If issues persist, check:
1. Project requirements are met (Python 3.10+)
2. All dependencies from `requirements.txt` installed
3. Running commands from project root directory