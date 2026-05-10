import re
from state import AgentState
from config import SCORING_THRESHOLDS

# Tamil Unicode block: U+0B80–U+0BFF
# Devanagari (Hindi) block: U+0900–U+097F
SCRIPT_PATTERNS = {
    "tamil":   re.compile(r"[\u0B80-\u0BFF]"),
    "hindi":   re.compile(r"[\u0900-\u097F]"),
    "telugu":  re.compile(r"[\u0C00-\u0C7F]"),
    "bengali": re.compile(r"[\u0980-\u09FF]"),
}


def detect_language_score(response: str, expected_language: str) -> float:
    """
    Check if the model actually responded in the correct script.
    1.0 = correct script used
    0.5 = romanized/transliterated (partial credit)
    0.0 = wrong language entirely
    """
    if not response.strip():
        return 0.0

    pattern = SCRIPT_PATTERNS.get(expected_language.lower())
    if pattern is None:
        return 1.0  # Unknown language — don't penalise

    native_chars = len(pattern.findall(response))
    total_chars   = len(response.strip())

    if native_chars == 0:
        # No native script at all — check if it's romanised (English letters only)
        if re.search(r"[a-zA-Z]", response):
            return 0.5  # Romanised — partial credit
        return 0.0
    elif native_chars / total_chars > 0.3:
        return 1.0   # Predominantly native script
    else:
        return 0.7   # Mixed — mostly native


def detect_hallucination(
    response: str,
    ground_truth: str,
    incorrect_answers: list,
    semantic_score: float,
    embedder,
) -> tuple[bool, str]:
    """
    Flag hallucination if:
    1. Response is semantically close to a known INCORRECT answer, OR
    2. Response sounds confident but is far from the ground truth
    """
    if not response.strip():
        return False, ""

    response_lower = response.lower()
    confidence_phrases = [
        "definitely", "certainly", "absolutely", "always", "never",
        "it is", "it's", "the answer is", "i can confirm", "without doubt",
        "நிச்சயமாக", "கண்டிப்பாக",  # Tamil confidence phrases
        "निश्चित रूप से", "बिल्कुल",  # Hindi confidence phrases
    ]
    sounds_confident = any(p in response_lower for p in confidence_phrases)

    # Check similarity against known incorrect answers
    if incorrect_answers and embedder:
        try:
            from sentence_transformers import util
            resp_emb = embedder.encode(response, convert_to_tensor=True)
            for wrong in incorrect_answers[:5]:
                if not wrong.strip():
                    continue
                wrong_emb = embedder.encode(wrong, convert_to_tensor=True)
                score = float(util.pytorch_cos_sim(resp_emb, wrong_emb)[0][0])
                if score > SCORING_THRESHOLDS["hallucination_similarity_to_incorrect"]:
                    return True, f"Response closely matches known incorrect answer (similarity: {score:.2f})"
        except Exception:
            pass

    # Confident but semantically wrong
    threshold = SCORING_THRESHOLDS["hallucination_confidence_with_low_accuracy"]
    if sounds_confident and semantic_score < threshold:
        return True, f"Response sounds confident but semantic score is low ({semantic_score:.2f})"

    return False, ""


def score_results(state: AgentState):
    print("--- Scoring Results ---")

    eval_results      = state.get("eval_results", [])
    scored_results    = state.get("scored_results", [])
    approved_questions = state.get("approved_questions", [])

    # Load multilingual embedding model
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer, util
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("Multilingual embedding model loaded.")
    except ImportError:
        print("sentence-transformers not installed — semantic scoring disabled.")

    questions = {q.get("id"): q for q in approved_questions}
    already_scored = {r.get("question_id") for r in scored_results}
    weights = {"semantic": 0.50, "language": 0.25, "cultural": 0.25}

    for res in eval_results:
        if res.get("error") or res.get("question_id") in already_scored:
            continue

        q = questions.get(res.get("question_id"))
        if not q:
            continue

        ground_truth       = q.get("ground_truth_native", "")
        correct_answers    = q.get("correct_answers", [])
        incorrect_answers  = q.get("incorrect_answers", [])
        expected_language  = q.get("language", "tamil")
        response           = res.get("final_answer", "")

        # ── Semantic score ────────────────────────────────────────────────
        semantic_score = 0.0
        if embedder and response.strip():
            try:
                from sentence_transformers import util
                resp_emb = embedder.encode(response, convert_to_tensor=True)
                gt_emb   = embedder.encode(ground_truth, convert_to_tensor=True)
                semantic_score = float(util.pytorch_cos_sim(resp_emb, gt_emb)[0][0])

                # Also check all correct answer variants — take the best
                for alt in correct_answers:
                    if alt.strip():
                        alt_emb = embedder.encode(alt, convert_to_tensor=True)
                        alt_score = float(util.pytorch_cos_sim(resp_emb, alt_emb)[0][0])
                        semantic_score = max(semantic_score, alt_score)

                semantic_score = round(max(0.0, min(1.0, semantic_score)), 4)
            except Exception as e:
                print(f"Embedding error: {e}")
                semantic_score = 0.0

        # ── Language score ────────────────────────────────────────────────
        language_score = detect_language_score(response, expected_language)

        # ── Cultural score (heuristic) ────────────────────────────────────
        # A high semantic score with correct language implies cultural accuracy.
        # We use semantic_score as a proxy until a dedicated cultural model is added.
        cultural_score = round(semantic_score * language_score, 4)

        # ── Hallucination ─────────────────────────────────────────────────
        is_hallucination, hallucination_reason = detect_hallucination(
            response, ground_truth, incorrect_answers, semantic_score, embedder
        )

        # ── Composite score ───────────────────────────────────────────────
        composite = round(
            semantic_score  * weights["semantic"] +
            language_score  * weights["language"] +
            cultural_score  * weights["cultural"],
            4
        )

        scored_results.append({
            "question_id":         res["question_id"],
            "model":               res["model"],
            "strategy":            res.get("strategy", "zero_shot"),
            "question":            q.get("question_native", ""),
            "ground_truth":        ground_truth,
            "response":            response,
            "semantic_score":      semantic_score,
            "language_score":      language_score,
            "cultural_score":      cultural_score,
            "composite_score":     composite,
            "is_correct":          semantic_score >= SCORING_THRESHOLDS["correct"],
            "is_hallucination":    is_hallucination,
            "hallucination_reason": hallucination_reason,
            "latency_ms":          res.get("latency_ms", 0),
        })

    print(f"Scored {len(scored_results)} results.")
    return {"scored_results": scored_results, "current_step": "scorer"}
