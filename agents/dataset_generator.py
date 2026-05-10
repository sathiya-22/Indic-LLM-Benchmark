import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from state import AgentState

SYSTEM_PROMPT = """You are an expert in Tamil and Indian languages with deep cultural knowledge. Your job is to generate original, high-quality benchmark questions for evaluating LLMs on Indian language understanding. Each question must be:
- Written natively in the target language (not translated from English)
- Culturally authentic and unambiguous
- Accompanied by a verified ground truth answer
- Categorised correctly
- Appropriate for the specified difficulty level

Never generate questions that can be answered by simply translating an English question. The questions must require genuine cultural or linguistic knowledge of India.
"""

def generate_dataset(state: AgentState):
    print(f"--- Generating Dataset for {state.get('language')} / {state.get('category')} ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_tokens=4096)
    
    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\n" +
        "Generate {count} questions for Language: {language}, Category: {category}, Difficulty: {difficulty}.\n" +
        "Avoid topics: {avoid_topics}\n" +
        "Output ONLY valid JSON with a 'questions' array containing objects with keys: id, language, category, difficulty, question_native, question_romanized, question_english, ground_truth_native, ground_truth_english, correct_answers (array), incorrect_answers (array), explanation, source, cultural_sensitivity (bool)."
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "count": state.get("count", 1),
        "language": state.get("language", "tamil"),
        "category": state.get("category", "cultural_knowledge"),
        "difficulty": state.get("difficulty", "medium"),
        "avoid_topics": state.get("avoid_topics", [])
    })
    
    try:
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
        generated_questions = json.loads(json_str).get("questions", [])
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        generated_questions = []
        
    current = state.get("generated_questions", [])
    return {"generated_questions": current + generated_questions, "current_step": "dataset_generator"}
