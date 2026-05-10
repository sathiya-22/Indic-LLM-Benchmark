import time
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from state import AgentState
from config import SUPPORTED_MODELS, PROMPTING_STRATEGIES

# ── Few-shot examples (Tamil & Hindi) ────────────────────────────────────────
FEW_SHOT_EXAMPLES = {
    "tamil": (
        "கேள்வி: தமிழ்நாட்டின் தலைநகரம் எது?\n"
        "பதில்: தமிழ்நாட்டின் தலைநகரம் சென்னை.\n\n"
        "கேள்வி: திருக்குறளை எழுதியவர் யார்?\n"
        "பதில்: திருக்குறளை திருவள்ளுவர் எழுதினார்.\n\n"
        "கேள்வி: பொங்கல் திருவிழா எந்த மாதத்தில் கொண்டாடப்படுகிறது?\n"
        "பதில்: பொங்கல் திருவிழா தை மாதத்தில் (ஜனவரி) கொண்டாடப்படுகிறது."
    ),
    "hindi": (
        "प्रश्न: भारत की राजधानी क्या है?\n"
        "उत्तर: भारत की राजधानी नई दिल्ली है।\n\n"
        "प्रश्न: महात्मा गांधी का जन्म कहाँ हुआ था?\n"
        "उत्तर: महात्मा गांधी का जन्म पोरबंदर, गुजरात में हुआ था।\n\n"
        "प्रश्न: दीपावली का त्योहार क्यों मनाया जाता है?\n"
        "उत्तर: दीपावली भगवान राम के अयोध्या लौटने की खुशी में मनाई जाती है।"
    ),
}

DEFAULT_EXAMPLES = "Q: What is 2+2?\nA: 4."


def get_llm(model_name: str):
    """Return the appropriate LangChain LLM for the given model name."""
    if "gemini" in model_name:
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    elif "gpt" in model_name:
        return ChatOpenAI(model=model_name, temperature=0.0)
    elif any(k in model_name for k in ["llama", "gemma", "mixtral"]):
        return ChatGroq(model=model_name, temperature=0.0)
    else:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


def extract_final_answer(raw: str, strategy: str) -> str:
    """For chain-of-thought, extract only the final answer."""
    if strategy == "chain_of_thought":
        markers = ["Final Answer:", "இறுதி பதில்:", "अंतिम उत्तर:"]
        for marker in markers:
            if marker in raw:
                return raw.split(marker)[-1].strip()
    return raw.strip()


def run_evaluations(state: AgentState):
    print("--- Running Evaluations ---")

    questions = state.get("approved_questions", [])
    eval_results = state.get("eval_results", [])

    if not questions:
        print("No approved questions to evaluate.")
        return {"eval_results": eval_results, "current_step": "eval_runner"}

    # ── Read models + strategies from state, fall back to sensible defaults ──
    models_to_test = state.get("models_to_test") or ["gemini-2.5-flash", "gemma-2-9b"]
    strategies     = state.get("strategies_to_test") or ["zero_shot", "few_shot", "chain_of_thought"]

    # Filter to only supported models
    models_to_test = [m for m in models_to_test if m in SUPPORTED_MODELS] or ["gemini-2.5-flash"]

    run_id = state.get("eval_run_id") or str(uuid.uuid4())
    already_run = {
        (r["question_id"], r["model"], r["strategy"])
        for r in eval_results
        if not r.get("error")
    }

    print(f"Models:     {models_to_test}")
    print(f"Strategies: {strategies}")
    print(f"Questions:  {len(questions)}")

    for q in questions:
        language = q.get("language", "tamil")
        examples = FEW_SHOT_EXAMPLES.get(language.lower(), DEFAULT_EXAMPLES)

        for model in models_to_test:
            for strategy in strategies:
                key = (q.get("id"), model, strategy)
                if key in already_run:
                    continue  # Skip already completed runs

                try:
                    llm = get_llm(model)
                    prompt_template = PROMPTING_STRATEGIES[strategy]
                    prompt = prompt_template.format(
                        language=language,
                        question=q.get("question_native", ""),
                        examples=examples,
                    )

                    start_time = time.time()
                    response   = llm.invoke(prompt)
                    latency_ms = (time.time() - start_time) * 1000

                    raw_response  = response.content or ""
                    final_answer  = extract_final_answer(raw_response, strategy)

                    usage = getattr(response, "response_metadata", {}).get("token_usage", {})

                    eval_results.append({
                        "question_id":       q.get("id"),
                        "model":             model,
                        "strategy":          strategy,
                        "prompt_sent":       prompt,
                        "raw_response":      raw_response,
                        "final_answer":      final_answer,
                        "latency_ms":        round(latency_ms, 2),
                        "prompt_tokens":     usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "error":             None,
                    })

                except Exception as e:
                    print(f"Error — {model}/{strategy}/{q.get('id')}: {e}")
                    eval_results.append({
                        "question_id": q.get("id"),
                        "model":       model,
                        "strategy":    strategy,
                        "error":       str(e),
                        "raw_response":   "",
                        "final_answer":   "",
                        "latency_ms":     0,
                        "prompt_tokens":  0,
                        "completion_tokens": 0,
                    })

                time.sleep(0.5)  # Respect rate limits

    print(f"Evaluation complete. Total results: {len(eval_results)}")
    return {
        "eval_results":  eval_results,
        "eval_run_id":   run_id,
        "current_step":  "eval_runner",
    }
