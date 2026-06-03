import os
import time
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
    with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)

#------------------Tools--------------
def search_tool(query):
    return f"[search] {query}"

def code_tool(task):
    return f"[code] executed: {task}"

def db_tool(query):
    return f"[db] retrieved: {query}"


#---------------Planner--------------
def planner_agent(state):
    prompt = f"""
    Create a shot plan for:
    {state['user_query']}
    """

    response = llm.invoke(prompt)
    state["plan"] = response.content if hasattr(response, "content") else str(response)
    write_text_file("planner.txt", state["plan"])
    return state

#--------------Worker----------------
def worker_agent(state):
    state["worker_calls"] += 1

    feedback = (
        state.get("behavior_feedback", "") + "\n" +
        state.get("reasoning_feedback", "") + "\n" +
        state.get("review_reason", "")
    )

    action_trace = []

    prompt = f"""
    you are an AI agent with tools:

    search_tool -> facts
    code_tool -> computation
    db_tool -> structured data

    RULES:
    - choose correct tool
    - one tool per step

    Format:
    Thought:
    Action:
    Action Input:

    Final Answer:
    <answer>

    user query: {state['user_query']}
    plan: {state['plan']}
    feedback: {feedback}
    """

    response = llm.invoke(prompt)
    output = response.content if hasattr(response, "content") else str(response)

    lines = output.splitlines()

    final_answer = ""

    current_action = None

    for line in lines:
        line_lower = line.lower()

        if line_lower.startswith("action:"):
            current_action = {"action": line.split(":", 1)[1].strip()}

        elif line_lower.startswith("action input:"):
            current_input = line.split(":", 1)[1].strip()

            if current_action:
                action_name = current_action.get("action", "")
                if "search" in action_name:
                    res = search_tool(current_input)
                elif "code" in action_name:
                    res = code_tool(current_input)
                elif "db" in action_name:
                    res = db_tool(current_input)
                else:
                    res = "unknown tool"
                
                action_trace.append({
                    "action": action_name,
                    "input": current_input,
                    "result": res
                })
        
        elif "final answer:" in line_lower:
            final_answer = line.split(":", 1)[1].strip()

    state["draft_response"] = final_answer
    state["action_trace"] = action_trace

    write_text_file(f"worker_output_{state['worker_calls']}.txt", output)
    return state

#--------------Behavior Eval--------------
def behavior_evaluator_agent(state):
    prompt = f"""
    Evaluate tool usage behavuor:

    Query:
    {state['user_query']}

    Action Trace:
    {state.get('action_trace', [])}

    score correct tool usage, efficiency, correctness

    Return:
    Score: 0-10
    Decision: approve OR revise
    Reason: short
    """

    res = llm.invoke(prompt).content
    res_str = str(res)
    state["behavior_decision"] = "approve" if "approve" in res_str.lower() else "revise"
    state["behavior_feedback"] = res_str

    write_text_file("behavior_eval.txt", res_str)
    return state

#-------------Reasoning Eval--------------
def reasoning_evaluator_agent(state):
    prompt = f"""
    Evaluate reasoning quality:

    Query:
    {state['user_query']}

    Return:
    Score: 0-10
    Decision: approve OR revise
    Reason: short
    """

    res = llm.invoke(prompt).content
    res_str = str(res)
    state["reasoning_decision"] = "approve" if "approve" in res_str.lower() else "revise"
    state["reasoning_feedback"] = res_str

    write_text_file("reasoning_eval.txt", res_str)
    return state

#-----------------Reviewer-------------
def reviewer_agent(state):
    prompt = f"""
    Evaluate final answer:

    {state['draft_response']}

    Return:
    Score: 0-10
    Decision: approve OR revise
    Reason: short
    """

    state["reviewer_calls"] += 1
    res = llm.invoke(prompt).content
    res_str = str(res)
    state["review_decision"] = "approve" if "approve" in res_str.lower() else "revise"
    state["review_reason"] = res_str

    write_text_file(f"review_{state['reviewer_calls']}.txt", res_str)
    return state

#--------------Router-------------------
def router(state):
    if(
        state.get("reviewer_decision") == "approve"
        and state.get("reasoning_decision") == "approve"
        and state.get("behavior_decision") == "approve"
    ) or state.get("revision_count", 0) >= 2:
        return "__end__"
    
    state["revision_count"] += 1
    return "worker"

#------------System Metrics------------
def computer_system_score(m):
    score = 10

    if m["execution_time"] > 10:
        score -=2
    if m["revision_count"] > 2:
        score -=2
    if m["total_calls"] > 6:
        score -=2
    if m["worker_calls"] > 3:
        score -= 2

    return max(score, 0)

#------------Graph---------------
workflow = StateGraph(dict)

workflow.add_node("planner", planner_agent)
workflow.add_node("worker", worker_agent)
workflow.add_node("behavior_eval", behavior_evaluator_agent)
workflow.add_node("reasoning_eval", reasoning_evaluator_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "worker")
workflow.add_edge("worker", "behavior_eval")
workflow.add_edge("behavior_eval", "reasoning_eval")
workflow.add_edge("reasoning_eval", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    router,
    {
        "worker": "worker",
        "__end__": END
    }
)

app = workflow.compile()

#-----------------RUN----------------
user_query = input("Enter query: ")

initial_state = {
    "user_query": user_query,
    "plan": "",
    "draft_response": "",
    "reasoning_trace": "",
    "action_trace": [],
    "review_reason": "",
    "reasoning_feedback": "",
    "behavior_feedback": "",
    "review_decision": "",
    "reasoning_decision": "",
    "behavior_decision": "",
    "worker_calls": 0,
    "reviewer_calls": 0,
    "revision_count": 0
}

#-------------Execution Timer-------------
start_time = time.time()
result = app.invoke(initial_state)
end_time = time.time()

#--------------System Metrics-------------
metrics = {
    "execution_time": end_time - start_time,
    "worker_calls": result.get("worker_calls", 0),
    "reviewer_calls": result.get("reviewer_calls", 0),
    "revision_count": result.get("revision_count", 0),
    "tool_calls": len(result.get("action_trace", []))
}

system_score = computer_system_score(metrics)

system_report = {
    **metrics,
    "system_score": system_score
}

write_text_file("system_eval.txt", str(system_report))

#-------------Output--------------
print("\n=== Final Output ===")
print(result.get("draft_response", ""))

print("\n=== System Evaluation ===")
print(system_report)