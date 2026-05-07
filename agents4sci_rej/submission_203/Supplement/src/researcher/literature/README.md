# Literature Query System

A comprehensive LLM-powered academic literature search system for the OneSim project. This system automatically extracts keywords from research queries using large language models and searches the ArXiv database to find relevant academic papers with abstracts, summaries, and relevance scoring.

## Features

- **LLM-Powered Keyword Extraction**: Intelligent keyword extraction from natural language research queries
- **ArXiv Integration**: Robust API client with rate limiting, error handling, and XML parsing
- **Multi-Strategy Search**: Parallel execution of different search strategies for comprehensive coverage  
- **Relevance Scoring**: Automatic relevance scoring and ranking of search results
- **Intelligent Caching**: Memory and disk-based caching for performance optimization
- **Multiple Output Formats**: Results available in JSON, Markdown, BibTeX, and CSV formats
- **REST API**: FastAPI-based HTTP endpoints for web service integration
- **OneSim Integration**: Seamless integration with OneSim's existing LLM infrastructure

## Architecture

```
literature/
├── models/              # Data models and schemas
│   ├── query_models.py     # Search queries and keyword results
│   └── literature_models.py # Papers, authors, and search results
├── config/              # Configuration management
│   └── literature_config.py # Settings and domain mappings
├── core/                # Core functionality
│   ├── arxiv_client.py     # ArXiv API client
│   ├── keyword_extractor.py # LLM keyword extraction
│   ├── query_processor.py  # Search orchestration
│   └── result_formatter.py # Output formatting
├── services/            # High-level services
│   ├── literature_service.py # Main service interface
│   └── cache_service.py    # Caching implementation
├── utils/               # Utility functions
│   └── text_utils.py      # Text processing utilities
├── api/                 # REST API endpoints
│   └── literature_api.py  # FastAPI routes
└── examples/            # Usage examples
    ├── basic_usage.py      # Basic usage examples
    ├── fastapi_server.py   # Web service example
    └── integration_with_onesim.py # OneSim integration
```

## Quick Start

### Basic Usage

```python
from researcher.literature.services.literature_service import LiteratureService

# Initialize the service
service = LiteratureService()
await service.initialize()

# Perform a literature search
result = await service.query_literature(
    query="transformer architecture for natural language processing",
    max_results=10,
    domain="computer science"
)

print(f"Found {len(result.papers)} papers")
for paper in result.papers[:3]:
    print(f"- {paper.title} (Relevance: {paper.relevance_score:.3f})")

await service.close()
```

### Web Service

```bash
# Run the FastAPI server
python src/researcher/literature/examples/fastapi_server.py --mode dev

# Query the API
curl -X POST "http://localhost:8000/api/literature/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "social dynamics in multi-agent systems",
    "domain": "computer science", 
    "max_results": 10
  }'
```

### OneSim Integration

```python
from researcher.literature.examples.integration_with_onesim import OnSimLiteratureIntegration

# Initialize integration
integration = OnSimLiteratureIntegration()
await integration.initialize()

# Get literature for a OneSim scenario
scenario_config = {
    "scenario_id": "social_dynamics_001",
    "description": "Study cultural evolution in multi-agent virtual society",
    "agents": {"cultural_background": ["western", "eastern"]},
    "metrics": ["cultural_homogeneity_index"]
}

literature_data = await integration.research_scenario_literature(
    scenario_config=scenario_config,
    max_papers=20
)

# Generate literature review report
report = await integration.generate_literature_report(
    scenario_config=scenario_config,
    output_file="literature_report.md"
)
```

## API Reference

### LiteratureService

Main service interface for literature queries.

#### Methods

- `query_literature(query, domain=None, max_results=20, mode=SearchMode.COMPREHENSIVE)`: Perform literature search
- `extract_keywords_only(text, domain=None, mode=SearchMode.COMPREHENSIVE)`: Extract keywords from text
- `batch_query(queries, domain=None, max_results_per_query=10)`: Process multiple queries
- `get_paper_details(arxiv_id, include_summary=True)`: Get detailed paper information
- `search_by_author(author_name, max_results=50)`: Search papers by author
- `get_recent_papers(domain=None, days=7, max_results=50)`: Get recent papers
- `get_service_stats()`: Get service statistics
- `clear_cache()`: Clear all cached data

### REST API Endpoints

- `POST /api/literature/query`: Query literature
- `POST /api/literature/extract-keywords`: Extract keywords from text
- `POST /api/literature/batch-query`: Batch query processing
- `GET /api/literature/paper/{arxiv_id}`: Get paper details
- `GET /api/literature/author/{author_name}`: Search by author
- `GET /api/literature/recent`: Get recent papers
- `GET /api/literature/status`: Service status
- `DELETE /api/literature/cache`: Clear cache
- `GET /api/literature/health`: Health check

## Configuration

The system uses a hierarchical configuration system with environment variables and YAML files.

### Environment Variables

```bash
# LLM Configuration
LITERATURE_LLM_MODEL=gpt-4
LITERATURE_LLM_TEMPERATURE=0.1
LITERATURE_LLM_MAX_TOKENS=2000

# ArXiv API
LITERATURE_ARXIV_RATE_LIMIT=3.0
LITERATURE_ARXIV_MAX_RESULTS=2000

# Caching
LITERATURE_CACHE_ENABLED=true
LITERATURE_CACHE_TTL_HOURS=24
LITERATURE_CACHE_DIR=./cache/literature

# API Server
LITERATURE_API_HOST=0.0.0.0
LITERATURE_API_PORT=8000
```

### Domain Mappings

The system includes built-in mappings for research domains to ArXiv categories:

- Computer Science → cs.*
- Physics → physics.*
- Mathematics → math.*
- Biology → q-bio.*
- Economics → econ.*
- Statistics → stat.*

## Search Modes

- `FOCUSED`: Fast, targeted search with primary keywords
- `COMPREHENSIVE`: Multi-strategy search with keyword expansion  
- `EXPLORATORY`: Broad search including related terms and concepts

## Performance Features

### Caching Strategy
- **Memory Cache**: Fast access for frequently used results
- **Disk Cache**: Persistent storage for long-term caching
- **Multi-level**: Keyword extraction, ArXiv responses, and complete results
- **TTL-based**: Configurable time-to-live for cache entries

### Rate Limiting
- Automatic rate limiting for ArXiv API compliance
- Configurable delays between requests
- Request queuing and retry mechanisms

### Parallel Processing
- Concurrent execution of multiple search strategies
- Asynchronous processing throughout the pipeline
- Efficient resource utilization

## Error Handling

The system implements comprehensive error handling with fallback mechanisms:

- **LLM Failures**: Fallback keyword extraction using rule-based methods
- **API Timeouts**: Automatic retry with exponential backoff
- **Network Issues**: Graceful degradation with partial results
- **Rate Limiting**: Automatic request pacing and queuing

## Testing

```bash
# Run basic usage examples
python src/researcher/literature/examples/basic_usage.py

# Start the web service
python src/researcher/literature/examples/fastapi_server.py

# Test OneSim integration
python src/researcher/literature/examples/integration_with_onesim.py
```

## Dependencies

### Core Dependencies
- `asyncio`: Asynchronous programming
- `aiohttp`: HTTP client for ArXiv API
- `lxml`: XML parsing for ArXiv responses
- `dataclasses`: Data structure definitions
- `pathlib`: File system operations

### Optional Dependencies
- `fastapi`: Web API framework
- `uvicorn`: ASGI server for FastAPI
- `pydantic`: Data validation for API

### OneSim Integration
- Uses OneSim's existing `ModelManager` for LLM access
- Compatible with OneSim's configuration system
- Integrates with OneSim's logging infrastructure

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update documentation for API changes
5. Use async/await patterns consistently

## License

This project is part of the OneSim framework and follows the same licensing terms.