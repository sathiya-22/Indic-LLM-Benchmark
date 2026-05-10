import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from state import AgentState

SYSTEM_PROMPT = """You are a rigorous quality reviewer for a multilingual AI benchmark. Your job is to review each question and flag issues before it enters the dataset. You must check:
1. CULTURAL ACCURACY
2. AMBIGUITY
3. DIFFICULTY CALIBRATION
4. LANGUAGE QUALITY
5. HALLUCINATION RISK
6. DUPLICATION

Output ONLY valid JSON with keys: question_id (string), approved (boolean), issues (array of strings), suggested_edits (object), rejection_reason (string), confidence (float 0.0-1.0)."""

def review_quality(state: AgentState):
    print("--- Reviewing Quality ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_tokens=1024)
    questions = state.get("generated_questions", [])
    
    approved_questions = state.get("approved_questions", [])
    needs_human_review = state.get("needs_human_review", [])
    review_feedback = state.get("review_feedback", [])
    
    # Process only new questions
    already_reviewed = {f.get("question_id") for f in review_feedback}
    new_questions = [q for q in questions if q.get("id") not in already_reviewed]
    
    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\nReview this question: {question}\n"
    )
    chain = prompt | llm
    
    for q in new_questions:
        response = chain.invoke({"question": json.dumps(q, ensure_ascii=False)})
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            else:
                content = content.strip()
            result = json.loads(content)
            
            review_feedback.append(result)
            
            if result.get("confidence", 1.0) < 0.7:
                needs_human_review.append(q)
            elif result.get("approved", False):
                approved_questions.append(q)
                
        except Exception as e:
            print(f"Review error for {q.get('id')}: {e}")
            
    return {
        "review_feedback": review_feedback,
        "approved_questions": approved_questions,
        "needs_human_review": needs_human_review,
        "current_step": "quality_reviewer"
    }
