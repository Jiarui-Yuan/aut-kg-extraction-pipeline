# AUT KG Extraction Pipeline

LLM-based extraction of structured knowledge from text into knowledge graphs.  
Research project at TU Berlin — Institut für Industrielle Automatisierungstechnik.

## Setup

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.ai)

```bash
git clone <repo-url>
cd aut-kg-extraction-pipeline
uv sync
uv pip install -e .
```

Pull an LLM:
```bash
ollama pull llama3.2
ollama serve
```

## Quick Start

Open `notebooks/01_extraction_basics.ipynb` and run the cells.

Or in Python:
```python
from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.entities import Tool

extractor = Extractor(model="llama3.2")
tools = extractor.extract_list("Wir verwenden das Fronius TPS 400i.", Tool)
```

## Project Structure

```
backend/
├── schemas/              # Pydantic models (what to extract)
│   ├── process_knowledge/    # Domain A: procedures, tools, steps
│   └── organizational_knowledge/  # Domain B: customers, suppliers
├── extraction/           # Text → Pydantic (LLM-based)
├── llm/                  # Shared LLM client (Ollama + Instructor)
├── graph/                # Pydantic → Knowledge Graph (insert, lookup, dedup)
├── training/             # Teacher-student finetuning pipeline
├── evaluation/           # P/R/F1 metrics, matching strategies
└── pipeline/             # End-to-end orchestration
```

## Using Different Models

```python
# Base model
extractor = Extractor(model="llama3.2")

# Larger model
extractor = Extractor(model="llama3.1:70b")

# Fine-tuned model (after training)
extractor = Extractor(model="my-finetuned-model")
```