# 🇮🇳 BharatEval — Indic LLM Benchmark

An open-source, multi-agent evaluation suite for benchmarking Large Language Models on Indian languages. Built with LangGraph, it automates the full pipeline from question generation to scoring and leaderboard publishing.

---

## What it does

Most LLM benchmarks are English-first. BharatEval fills that gap by:

- **Generating** culturally grounded benchmark questions in Tamil and Hindi across 7 categories
- **Reviewing** question quality automatically using an AI reviewer agent
- **Evaluating** multiple LLMs side-by-side using zero-shot, few-shot, and chain-of-thought prompting
- **Scoring** responses on semantic accuracy, language correctness, and cultural accuracy
- **Publishing** results to a live Streamlit leaderboard

---

## Pipeline

```
Dataset Generator → Quality Reviewer → Eval Runner → Scorer → Report Generator
                          ↓
                   (Human review queue if needed)
```

Built on **LangGraph** — each stage is a node in a stateful graph, with conditional routing based on review outcomes.

---

## Supported Models

| Model | Provider |
|---|---|
| `gemini-2.5-flash` | Google |
| `gpt-4o` | OpenAI |
| `llama-3.3-70b` | Groq |
| `gemma-2-9b` | Groq |
| `mixtral-8x7b` | Groq |
| `ai4bharat-indic-bert` | AI4Bharat |

---

## Categories

| Category | Weight |
|---|---|
| Cultural Knowledge | 20% |
| Reasoning (Native) | 15% |
| Factual India | 15% |
| Language Understanding | 15% |
| Code Switching | 15% |
| Transliteration | 10% |
| Bias Detection | 10% |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/sathiya-22/Indic-LLM-Benchmark.git
cd Indic-LLM-Benchmark

# 2. Install dependencies
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env and add your keys

# 4. Run the pipeline
python main.py

# 5. Launch the leaderboard
streamlit run app.py
```

---

## Configuration

Edit `main.py` to change the run parameters:

```python
initial_state = {
    "language": "tamil",          # "tamil" or "hindi"
    "category": "cultural_knowledge",
    "difficulty": "medium",        # "easy", "medium", "hard"
    "count": 50,                   # number of questions to generate
    "models_to_test": ["gemini-2.5-flash", "gpt-4o"],
    "strategies_to_test": ["zero_shot", "chain_of_thought"],
}
```

Full configuration options are in `config.py`.

---

## Environment Variables

Create a `.env` file from `.env.example`:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

At least one key is required. Models are skipped if their provider key is missing.

---

## Project Structure

```
BharatEval/
├── agents/
│   ├── dataset_generator.py   # Generates benchmark questions
│   ├── quality_reviewer.py    # Reviews and approves questions
│   ├── eval_runner.py         # Runs LLMs against the benchmark
│   ├── scorer.py              # Scores responses
│   └── report_generator.py   # Generates summary reports
├── app.py                     # Streamlit leaderboard
├── graph.py                   # LangGraph workflow definition
├── state.py                   # Shared agent state schema
├── config.py                  # Dataset and model configuration
├── main.py                    # Entry point
├── requirements.txt
└── .env.example
```

---

## Scoring

Each response is scored across three dimensions:

| Dimension | Weight | Description |
|---|---|---|
| Semantic Accuracy | 50% | Is the answer correct? |
| Language Correctness | 25% | Is it in the right language and script? |
| Cultural Accuracy | 25% | Is it culturally appropriate for the Indian context? |

Hallucination detection flags responses that show high confidence but low accuracy.

---

## Leaderboard

The live leaderboard is available at **bharateval.streamlit.app** (run `streamlit run app.py` locally).

Results are stored in `data/results_database.db` and updated after each pipeline run.

---

## License

Dataset released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code under MIT.

---

## Acknowledgements

Built on top of [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and the [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library).
