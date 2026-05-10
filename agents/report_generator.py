import json
import sqlite3
import os
import pandas as pd
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from state import AgentState

# Always resolve paths relative to the project root — never hardcode
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH      = str(DATA_DIR / "results_database.db")
LEADERBOARD_PATH  = str(DATA_DIR / "leaderboard.json")
BLOG_POST_PATH    = str(DATA_DIR / "blog_post_draft.md")

SYSTEM_PROMPT = """You are a technical writer and data analyst specialising in AI evaluation. Your job is to take raw evaluation results and turn them into compelling, accurate insights. You must:
- Find the single most surprising or important finding
- Be honest about model failures without sensationalising
- Write findings that a non-technical reader can understand
- Back every claim with a specific number from the results
- Never overstate what the data shows

Output ONLY valid JSON matching the schema requirements."""

def generate_report(state: AgentState):
    print("--- Generating Report ---")
    
    scored_results = state.get("scored_results", [])
    
    if not scored_results:
        return {"current_step": "report_generator"}
        
    df = pd.DataFrame(scored_results)
    
    overall_scores = df.groupby("model")["composite_score"].mean().to_dict()
    hallucination_rates = df.groupby("model")["is_hallucination"].mean().to_dict()
    
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("evaluation_results", conn, if_exists="replace", index=False)
    conn.close()
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_tokens=4096)
    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\n" +
        "Scores: {scores}\nHallucinations: {hallucinations}\nTotal Questions: {total}\n" +
        "Output ONLY valid JSON with keys: headline_finding (string), leaderboard (array), category_breakdown (object), worst_failures (array), blog_post_draft (object with title, hook, methodology, key_findings, example_failures, conclusion, cta)."
    )
    
    chain = prompt | llm
    try:
        response = chain.invoke({
            "scores": json.dumps(overall_scores),
            "hallucinations": json.dumps(hallucination_rates),
            "total": len(scored_results)
        })
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        else:
            content = content.strip()
        report_data = json.loads(content)
    except Exception as e:
        print(f"Report generation error: {e}")
        report_data = {"error": str(e)}
        
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(report_data.get("leaderboard", []), f, indent=2)
        
    if "blog_post_draft" in report_data:
        with open(BLOG_POST_PATH, "w") as f:
            blog = report_data["blog_post_draft"]
            f.write(f"# {blog.get('title', 'BharatEval Results')}\n\n")
            f.write(f"{blog.get('hook', '')}\n\n")
            f.write(f"## Methodology\n{blog.get('methodology', '')}\n\n")
            f.write("## Key Findings\n")
            for k in blog.get("key_findings", []):
                f.write(f"- {k}\n")
            f.write(f"\n## Example Failures\n{blog.get('example_failures', '')}\n\n")
            f.write(f"## Conclusion\n{blog.get('conclusion', '')}\n\n")
            f.write(f"*{blog.get('cta', '')}*\n")
            
    return {"report_data": report_data, "current_step": "report_generator"}
