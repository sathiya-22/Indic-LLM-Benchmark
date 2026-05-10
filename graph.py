from langgraph.graph import StateGraph, END
from state import AgentState
from agents.dataset_generator import generate_dataset
from agents.quality_reviewer import review_quality
from agents.eval_runner import run_evaluations
from agents.scorer import score_results
from agents.report_generator import generate_report

def route_after_review(state: AgentState):
    """Routing logic after quality review."""
    if len(state.get("needs_human_review", [])) > 0:
        return "human_review" # In reality, we'd use a checkpointer here to pause
    elif len(state.get("approved_questions", [])) == 0:
        return "dataset_generator"
    return "eval_runner"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("dataset_generator", generate_dataset)
    workflow.add_node("quality_reviewer", review_quality)
    workflow.add_node("eval_runner", run_evaluations)
    workflow.add_node("scorer", score_results)
    workflow.add_node("report_generator", generate_report)
    
    workflow.set_entry_point("dataset_generator")
    workflow.add_edge("dataset_generator", "quality_reviewer")
    
    workflow.add_conditional_edges(
        "quality_reviewer",
        route_after_review,
        {
            "human_review": END, 
            "dataset_generator": "dataset_generator",
            "eval_runner": "eval_runner"
        }
    )
    
    workflow.add_edge("eval_runner", "scorer")
    workflow.add_edge("scorer", "report_generator")
    workflow.add_edge("report_generator", END)
    
    return workflow.compile()
