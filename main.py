import os
from dotenv import load_dotenv
from graph import build_graph

# Load environment variables
load_dotenv()

def run_pipeline():
    print("Initializing BharatEval Pipeline...")
    
    # Check for minimal API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY is not set. Execution will fail.")
        
    app = build_graph()
    
    initial_state = {
        "language": "tamil",
        "category": "cultural_knowledge",
        "difficulty": "medium",
        "count": 2,  # Increase to 50+ for a real run
        "avoid_topics": [],
        "generated_questions": [],
        "review_feedback": [],
        "approved_questions": [],
        "needs_human_review": [],
        "eval_run_id": "",
        "eval_results": [],
        "models_to_test": ["gemini-2.5-flash"],
        "strategies_to_test": ["zero_shot", "chain_of_thought"],  # or "few_shot"
        "scored_results": [],
        "report_data": {},
        "current_step": "init",
        "errors": []
    }
    
    print("Running Pipeline...")
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n--- Pipeline Completed ---")
    print(f"Total Questions Generated: {len(final_state.get('generated_questions', []))}")
    print(f"Total Approved: {len(final_state.get('approved_questions', []))}")
    print(f"Total Needs Human Review: {len(final_state.get('needs_human_review', []))}")
    
    report_data = final_state.get("report_data", {})
    if report_data:
        print("\nHeadline Finding:")
        print(report_data.get("headline_finding", "No finding generated."))
        print("\nCheck the data/ folder for detailed reports and leaderboards.")

if __name__ == "__main__":
    run_pipeline()
