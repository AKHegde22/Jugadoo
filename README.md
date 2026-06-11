# JugaadReasoning-1K

> **Constraint-Satisfying Resource-Substitution Benchmark for LLM Evaluation**

A 1,000-problem benchmark designed to test LLMs' capacity for *affordance-based physical resource substitution* under severe real-world constraints — financial, environmental, and infrastructural.

## Motivation

Current LLM benchmarks (GSM8K, ARC, MMLU) assume resource abundance, modern infrastructure, and accessible supply chains. JugaadReasoning-1K exposes a systemic cognitive gap: the inability to map everyday objects to alternative physical functionalities when standard tools are missing.

## Pipeline Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 1. SCRAPE SEEDS │ →  │ 2. EXTRACT TUPLES│ →  │ 3. MUTATE ×10   │
│   NIF, HoneyBee │    │   LLM-assisted   │    │  100 → 1,000    │
│   YouTube       │    │   structuring    │    │  constraint     │
└─────────────────┘    └──────────────────┘    │  matrix         │
                                                └────────┬────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────▼────────┐
│ 7. PLOT RESULTS │ ←  │ 6. JUDGE + SCORE │ ←  │ 5. RUN BENCHMARK │
│   3 pub-quality │    │   Keyword Guard  │    │   9 models       │
│   figures       │    │   LLM-as-Judge   │    │   MCQ + OpenGen  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                         ▲
                                                ┌────────┴────────┐
                                                │ 4. FORMAT       │
                                                │   MCQ + OpenGen │
                                                │   with rubrics  │
                                                └─────────────────┘
```

## Quick Start

### 1. Setup

```bash
# Clone and create virtual environment
git clone <repo-url> && cd Jugadoo
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run the Pipeline

```bash
# Step 1: Scrape raw innovation data
python scripts/01_scrape_seeds.py --sources nif_pdf

# Step 2: Extract structured seed tuples (LLM-assisted)
python scripts/02_extract_seed_tuples.py

# Step 3: Generate 1,000-row mutation matrix
python scripts/03_mutate_seeds.py

# Step 4: Format into MCQ + Open Generation datasets
python scripts/04_format_dataset.py

# Step 5: Run benchmark (use --dry-run first for cost estimation)
python scripts/05_run_benchmark.py --dry-run
python scripts/05_run_benchmark.py

# Step 6: Run judge verification protocol
python scripts/06_run_judge.py

# Step 7: Generate publication plots
python scripts/07_generate_plots.py
```

## Dataset Schema

Each benchmark problem follows this structure:

```json
{
  "problem_id": "JR-1K-001-01",
  "domain": "agriculture",
  "prompt_context": "A farmer in Maharashtra needs to irrigate 50 saplings...",
  "applied_constraints": {
    "budget": "₹0 budget",
    "environment": "45°C Heatwave",
    "infrastructure": "Total Grid Outage"
  },
  "available_inventory": ["50 discarded IV bottles with tubing...", "bamboo stakes", ...],
  "mcq_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "ground_truth_option": "C",
  "ground_truth_synthesis_rubric": {
    "essential_keywords": ["IV bottle", "drip", "clamp"],
    "forbidden_keywords": ["buy", "online", "motor"],
    "required_physical_mechanism": "Gravity-fed drip via roller clamp..."
  }
}
```

## Evaluation

- **MCQ**: Zero-shot, temperature=0.0, accuracy metric
- **Open Generation**: Two-stage verification:
  1. Keyword Guard (auto-fail on forbidden terms)
  2. LLM-as-a-Judge (3-point rubric: constraint adherence, inventory utilization, physical viability)
- **Judge Calibration**: Cohen's Kappa ≥ 0.80 on 100-sample human annotation subset

## Target Models

| Tier | Models |
|------|--------|
| Frontier | GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet |
| Open-Weights | Llama-3-70B, Mistral-Large, DeepSeek-V3 |
| Indic-Native | Krutrim, Tamil-Llama, Airavata |

## Data Sources

- [National Innovation Foundation (NIF) India](https://nif.org.in/) — Award Book PDFs
- [Honey Bee Network / SRISTI](https://sristi.org/) — Newsletter archives
- YouTube grassroots innovation channels (via Data API v3)

## License

Apache-2.0
