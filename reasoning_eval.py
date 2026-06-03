import os
import logging

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# ----------------Logging setup----------------
LOG_DIR="logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'execution.log'),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)

logger = logging.getLogger(__name__)

def write_text_file(filename:str, content:str):
    filepath=os.path.join(LOG_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# ----------------Planner----------------
def planner_agent(state):
    logger.info(f"Planner agent started")
    
    prompt = f"""
    You are a planning agent in a multi-agent AI system. 
    Create a short actionable plan for the worker agent to answer the user's query.
    User Query: {state['user_query']}
    """

    response = llm.invoke(prompt)
    plan = response.content if hasattr(response, 'content') else str(response)
    state['plan'] = plan
    write_text_file('planner_output.txt', plan)

    return state

# ----------------Worker----------------
def worker_agent(state):
    state['worker_calls']+=1
    feedback = (
        state.get('review_reason', "") + "\n" +
        state.get('reason_feedback', "")
    )

    logger.info(f"worker call #{state['worker_calls']}")

    prompt = f"""
    You are a worker agent.
    Solve the task step-by-step internally.
    Return in this format:

    FINAL ANSWER:
    <answer>

    REASONING TRACE:
    - Step 1:
    - Step 2:
    - Step 3:

    user query: {state['user_query']}
    plan: {state['plan']}
    feedback: {feedback}
    """

    response = llm.invoke(prompt)
    output = response.content if hasattr(response, 'content') else str(response)

    # Parse Output
    final_answer = ""
    reasoning_trace = ""

    if "REASONING TRACE:" in output:
        parts = output.split("REASONING TRACE:")
        final_answer = parts[0].replace("FINAL ANSWER:", "").strip()
        reasoning_trace = parts[1].strip()
    else:
        final_answer = output

    state['draft_response'] = final_answer
    state['reasoning_trace'] = reasoning_trace

    write_text_file(f'worker_output_{state["worker_calls"]}.txt', output)
    return state

# ----------------REASONING EVALUATOR----------------
def reasoning_evaluator_agent(state):
    logger.info(f"Reasoning Evaluator started")

    prompt = f"""
    You are a reasoning evaluator agent.
    Evaluate the reasoning trace for:
    - Logical concsistency
    - Missing steps
    - Incorrect assumptions
    - Hallucinations
    - Coherence

    user query: {state['user_query']}

    Reasoning Trace:
    {state.get('reasoning_trace', "")}

    Return EXACTLY:
    Score: <0-10>
    Decision: approve OR revise
    Reason: <short explanation>
    """

    response = llm.invoke(prompt)
    raw_output = response.content.strip() if hasattr(response, 'content') else str(response)

    decision = "approve" if "approve" in raw_output.lower() else "revise"

    score_line = next((line for line in raw_output.splitlines() if line.lower().startswith("score:")), "")
    score = score_line.replace("Score:", "").strip() if score_line else "N/A"

    reason_line = next((line for line in raw_output.splitlines() if line.lower().startswith("reason:")), "")

    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No Reason"

    state['reasoning_score'] = score
    state['reasoning_decision'] = decision
    state['reason_feedback'] = reason

    write_text_file("reasoning_evaluator.txt", raw_output)
    return state

# ----------------Output Reviewer----------------
def reviewer_agent(state):
    state['reviewer_calls']+=1
    
    prompt = f"""
    You are a strict reviewer agent.
    check:
    - Examples
    - Implementation details
    - Clarity
    - Actionability

    user query: {state['user_query']}
    answer: {state['draft_response']}

    Return EXACTLY:
    Decision: approve OR revise
    Reason: <short reason>
    """

    response = llm.invoke(prompt)
    raw_output = response.content.strip() if hasattr(response, 'content') else str(response)
    decision = "approve" if "approve" in raw_output.lower() else "revise"
    
    reason_line = next((line for line in raw_output.splitlines() if line.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No Reason"
    
    state['review_decision'] = decision
    state['review_reason'] = reason

    write_text_file(f'reviewer_output_{state["reviewer_calls"]}.txt', raw_output)
    return state

# ----------------Router----------------
def review_router(state):
    if (
        state.get('review_decision') == "approve" 
        and state.get('reasoning_decision') == "approve"
    ) or state.get('revision_count', 0) >= 2:
        return "__end__"
    
    state['revision_count'] += 1
    return "worker_agent"

# ----------------Graph Setup----------------
workflow = StateGraph(dict)

workflow.add_node("planner_agent", planner_agent)
workflow.add_node("worker_agent", worker_agent)
workflow.add_node("reasoning_evaluator_agent", reasoning_evaluator_agent)
workflow.add_node("reviewer_agent", reviewer_agent)

workflow.set_entry_point("planner_agent")
workflow.add_edge("planner_agent", "worker_agent")
workflow.add_edge("worker_agent", "reasoning_evaluator_agent")
workflow.add_edge("reasoning_evaluator_agent", "reviewer_agent")

workflow.add_conditional_edges("reviewer_agent", review_router,
                              {
                                  "worker_agent": "worker_agent",
                                  "__end__": END
                              }
)

app = workflow.compile()

# -----------------Graph Image----------------
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
except Exception as e:
    logger.error(f"Graph generating failed: {e}")

# ------------------RUN------------------
user_query = input("Enter your query: ")

initial_state = {
    "user_query": user_query,
    "plan": "",
    "draft_response": "",
    "reasoning_trace": "",
    "review_reason": "",
    "review_decision": "",
    "reasoning_decision": "",
    "worker_calls": 0,
    "reviewer_calls": 0,
    "revision_count": 0
}

result = app.invoke(initial_state)
final_output = result.get("draft_response", "")
write_text_file('final_output.txt', final_output)

print("\n=== Final Output ===")
print(final_output)
