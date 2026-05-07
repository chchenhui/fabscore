# AutoSurvey Baseline - Modifications Summary

## Overview
This directory contains the AutoSurvey baseline system with our modifications for the research comparison.

## Original AutoSurvey Files
- `main.py` - Original AutoSurvey entry point
- `evaluation.py` - Original evaluation script  
- `src/agents/` - Original agent implementations (outline_writer.py, writer.py, judge.py)
- `src/prompt.py` - Original prompt templates
- `src/database.py` - Original database interface
- `requirements.txt` - Original dependencies
- `README.md` - Original documentation

## Our Modifications and Additions

### Core Enhancements
- `run_local_survey_robust.py` - **New**: Robust survey runner with retry logic and improved error handling
- `src/model_robust.py` - **New**: Enhanced model interface with robust API handling
- `src/model_openai_robust.py` - **New**: OpenAI-specific robust model implementation
- `src/database_chunked.py` - **New**: Support for chunked database operations

### Integration Scripts (`scripts/`)
- `convert_pasa_to_autosurvey.py` - **New**: Convert PASA paper format to AutoSurvey format
- `convert_pasa_full.py` - **New**: Full PASA conversion pipeline
- `convert_pasa_chunked.py` - **New**: Chunked conversion for large datasets
- `build_embeddings_pasa.py` - **New**: Build embeddings for PASA papers
- `build_embeddings.py` - **New**: General embedding builder
- `test_setup.py` - **New**: Setup validation tests
- `test_openai_client.py` - **New**: OpenAI client testing

### Key Improvements Made
1. **Robust Error Handling**: Added retry logic and better error recovery
2. **PASA Integration**: Scripts to work with PASA paper format
3. **Chunked Processing**: Support for large-scale paper collections
4. **Local LLM Support**: Enhanced model interfaces for local API endpoints
5. **Improved Parsing**: Better handling of LLM response formats

## Usage for Research Comparison
The modified AutoSurvey system was used as the baseline in our multi-agent vs single-agent comparison:

- **Papers Processed**: 75-100 per topic
- **Performance**: 4.77/10 average score (C grade)
- **Key Limitation**: Poor synthesis and critical analysis capabilities
- **Comparison**: Significantly outperformed by our multi-agent system (8.17/10)

## Running the System
```bash
# Use the robust version for better reliability
python run_local_survey_robust.py --topic "Your Research Topic"

# Or use original interface
python main.py --query "Your Research Query"
```

## Dependencies
See `requirements.txt` for the full dependency list. Key additions include:
- Enhanced API client libraries
- Robust retry mechanisms
- PASA format processing tools

---
*This baseline was used to generate the AutoSurvey results in our research comparison.*