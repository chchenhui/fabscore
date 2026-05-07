# Reproducibility Statement

To enable full reproduction of our experimental results, we provide the complete experimental configuration and technical implementation details below.

## Code and Data Availability

The complete codebase and dataset are publicly available as open-source software at: [https://anonymous.4open.science/r/Simploy-7818](https://anonymous.4open.science/r/Simploy-7818)

The repository includes:
- Source code
- Configuration files
- Test suites
- Documentation
- Analysis scripts

## Simulation Parameters

- **Population**: 200 freelancer agents, 30 client agents
- **Duration**: 100 rounds per simulation
- **Replication**: 20 independent runs per configuration
- **Job posting cooldowns**: 2-7 turns (randomly assigned)
- **Jobs shown per freelancer per round**: 5 (randomly selected)
- **Maximum bids per freelancer per round**: 3
- **Maximum active jobs per freelancer**: 3
- **Agent bid constraints**: 1-5 bids per round, forcing strategic selectivity

## Agent Configuration

- **Random freelancers**: 5% bid probability per job shown
- **Random clients**: 50% acceptance probability for received bids
- **Random bid values**: Uniform distribution between 50-150% of job budget
- **Reflection probability**: 5% per turn for both freelancers and clients
- **Reputation system**: Four-tier progression (New → Established → Expert → Elite)

## Technical Implementation

- **LLM Model**: GPT-4o-mini for all agent reasoning, persona generation, and job posting
- **Software**: Python 3.9+, NumPy 1.21+, SciPy 1.7+, Matplotlib 3.4+, OpenAI API
- **LLM Parameters**: temperature and max_tokens settings for different agent operations are detailed in the source code


## Experimental Configurations

- **LLM-F + LLM-C**: Both freelancers and clients use GPT-4o-mini reasoning
- **LLM-F + LLM-C (w/o Reflections)**: Same as above but without reflection mechanisms
- **Rand-F + Rand-C**: Both agent types use probabilistic decision-making
- **LLM-F + Rand-C**: GPT-4o-mini freelancers with probabilistic clients
- **Rand-F + LLM-C**: Probabilistic freelancers with GPT-4o-mini clients

## Computational Resources

All experiments were conducted on an AWS EC2 instance with 48 CPU cores and 373 GiB of RAM. Execution times varied significantly by configuration due to the computational requirements of different agent types:

- **LLM-F + LLM-C configuration**: Approximately 50 minutes per run using 30 concurrent workers
- **Primary bottleneck**: Network latency for API calls rather than local computational resources
- **Random agent configurations** (Rand-F + Rand-C, LLM-F + Rand-C, Rand-F + LLM-C): Completed substantially faster as they do not or partially require external API calls

## Running the Experiments

1. Clone the repository from the provided URL
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API keys and settings as described in `config/config.example.py`
4. Run experiments: `python run_experiments.py`
5. Generate analysis: `python multi_run_analysis.py`

For detailed setup instructions, see `SETUP_GUIDE.md` in the repository.
