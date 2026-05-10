import os
from typing import List, Dict, Any

# Dataset Configuration
DATASET_CONFIG = {
    "target_size": 500,
    "languages": ["tamil", "hindi"],
    "category_distribution": {
        "cultural_knowledge": 0.20,
        "reasoning_native": 0.15,
        "factual_india": 0.15,
        "language_understanding": 0.15,
        "code_switching": 0.15,
        "transliteration": 0.10,
        "bias_detection": 0.10
    },
    "difficulty_distribution": {
        "easy": 0.30,
        "medium": 0.50,
        "hard": 0.20
    },
    "storage": "huggingface_datasets",
    "dataset_name": "sathiya-22/BharatEval",
    "license": "cc-by-4.0"
}

# Supported Models
SUPPORTED_MODELS = [
    "gpt-4o",
    "gemini-2.5-flash",
    "llama-3.3-70b",
    "gemma-2-9b",
    "mixtral-8x7b",
    "ai4bharat-indic-bert"
]

# Prompting Strategies
PROMPTING_STRATEGIES = {
    "zero_shot": "Answer the following question accurately in {language}. If unsure, say so rather than guessing.\n\nQuestion: {question}\n\nAnswer:",
    "few_shot": "Answer questions accurately in {language}, following these examples:\n\n{examples}\n\nQuestion: {question}\n\nAnswer:",
    "chain_of_thought": "Answer the following question in {language}. Think through it step by step, then give your Final Answer on a new line.\n\nQuestion: {question}\n\nLet me think step by step:"
}

# Scoring Thresholds
SCORING_THRESHOLDS = {
    "correct": 0.60,
    "hallucination_similarity_to_incorrect": 0.65,
    "hallucination_confidence_with_low_accuracy": 0.30
}

# Leaderboard Config
LEADERBOARD_CONFIG = {
    "update_frequency": "on_new_run",
    "hosting": "streamlit_cloud",
    "url": "bharateval.streamlit.app",
    "composite_score_weights": {
        "semantic_accuracy": 0.50,
        "language_correctness": 0.25,
        "cultural_accuracy": 0.25
    }
}
