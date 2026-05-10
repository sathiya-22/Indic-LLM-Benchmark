from typing import TypedDict, Annotated, List, Dict, Any, Optional

class AgentState(TypedDict):
    # Dataset generation
    language: str
    category: str
    difficulty: str
    count: int
    avoid_topics: List[str]
    
    generated_questions: List[Dict[str, Any]]
    
    # Quality Review
    review_feedback: List[Dict[str, Any]]
    approved_questions: List[Dict[str, Any]]
    needs_human_review: List[Dict[str, Any]]
    
    # Evaluation Runner
    eval_run_id: str
    eval_results: List[Dict[str, Any]]
    models_to_test: Optional[List[str]]       # e.g. ["gemini-1.5-pro", "gemma-2-9b"]
    strategies_to_test: Optional[List[str]]   # e.g. ["zero_shot", "few_shot", "chain_of_thought"]
    
    # Scorer
    scored_results: List[Dict[str, Any]]
    
    # Report Generator
    report_data: Dict[str, Any]
    
    # Control flow variables
    current_step: str
    errors: List[str]
