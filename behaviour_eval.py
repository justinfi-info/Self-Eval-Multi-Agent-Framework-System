import os
import logging

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import groq

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
llm = ChatGroq(model=MODEL_NAME)

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


def invoke_llm(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except groq.RateLimitError as e:
        logger.error("LLM rate limit error", exc_info=True)
        raise RuntimeError(
            "LLM rate limit exceeded. Please wait or switch to a different model."
        ) from e
    except Exception as e:
        logger.error("LLM invocation failed", exc_info=True)
        raise

# ----------------Tools----------------
def search_tool(query):
    return f"[Search results] info about: {query}"

def code_tool(task):
    return f"[Code execution] solved: {task}"

def db_tool(query):
    return f"[DB results] retrieved structured data for: {query}"

# ----------------Planner----------------
def planner_agent(state):
    
    prompt = f"""
    You are a planning agent.
    Create a short actionable plan for the worker agent to answer the user's query.
    User Query: {state['user_query']}
    """

    response = invoke_llm(prompt)
    state['plan'] = response
    
    write_text_file('planner_output.txt', state['plan'])
    return state

# ----------------Worker [Behavior]----------------
def worker_agent(state):
    state['worker_calls']+=1
    feedback = (
        state.get('behavior_feedback', "") + "\n" +
        state.get('reasoning_feedback', "") + "\n" +
        state.get('review_reason', "")
    )

    action_trace = []

    prompt = f"""
    You are an AI agent with tools:
    - Search tool: use this to search the web for information.
    - Code tool: use this to execute code for tasks like calculations, data processing, etc.
    - DB tool: use this to query a database for structured information.

    STRICT RULES:
    - Choose correct tool
    - Do not hallucinate if tool needed
    - One tool per step 

    Format:
    Thought:
    Action:
    Action Input:

    Repeat if needed.

    Then:
    FINAL ANSWER:
    <answer>

    REASONING TRACE:
    - Step 1:
    - Step 2:

    user query: {state['user_query']}
    plan: {state['plan']}
    feedback: {feedback}
    """

    output = invoke_llm(prompt)

    lines = output.splitlines()

    final_answer = ""
    reasoning_trace = ""
    current_action = None

    for i, line in enumerate(lines):
        line_lower = line.lower()

        if line_lower.startswith("action:"):
            current_action = {"action": line.split(":", 1)[1].strip()}

        elif line_lower.startswith("action input:"):
            current_input = line.split(":", 1)[1].strip()

            if current_action:
                action_name = current_action.get("action", "").lower()

                if "search_tool" in action_name:
                    result = search_tool(current_input)
                elif "code_tool" in action_name:
                    result = code_tool(current_input)
                elif "db_tool" in action_name:
                    result = db_tool(current_input)
                else:
                    result = "Unknown tool"

                action_trace.append({
                    "action": current_action["action"],
                    "input": current_input,
                    "result": result
                })

        elif line_lower.startswith("final answer:"):
            final_answer = line.split(":", 1)[1].strip()

        elif line_lower.startswith("reasoning trace:"):
            reasoning_trace = "\n".join(lines[i+1:]).strip()

    state['draft_response'] = final_answer
    state['reasoning_trace'] = reasoning_trace
    state['action_trace'] = action_trace

    write_text_file(f"worker_output_{state['worker_calls']}.txt", output)
    write_text_file(f"action_trace_{state['worker_calls']}.txt", str(action_trace))

    return state

# ----------------Behavior Evaluator----------------
def behavior_evaluator_agent(state):

    prompt = f"""
    You are a behavior evaluator agent.
    Focus on:
    - Correct tool selection
    - Missing tool usage
    - Unnecessary actions
    - Efficiency
    - Logical order

    Tools:
    - search_tool -> factual
    - code_tool -> computational
    - db_tool -> structured

    User Query: {state['user_query']}

    Action Trace: {state.get('action_trace', [])}

    Rules:
    - Wrong tool -> major penalty
    - No tool when needed -> penalty
    - Extra steps -> penalty

    Return EXACTLY:
    Score: <0-10>
    Decision: approve OR revise
    Reason: <short explanation>
    """

    raw = invoke_llm(prompt).strip()

    decision = "approve" if "approve" in raw.lower() else "revise"
    reason_line = next((line for line in raw.splitlines() if line.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No Reason"

    state['behavior_decision'] = decision
    state['behavior_feedback'] = reason

    write_text_file("behavior_eval.txt", raw)
    return state

# ----------------Reasoning Evaluator----------------
def reasoning_evaluator_agent(state):
    prompt = f"""
    Evaluate reason.

    Query: {state['user_query']}
    Reasoning Trace: {state.get('reasoning_trace', "")}

    Return EXACTLY:
    Score: <0-10>
    Decision: approve OR revise
    Reason: <short explanation>
    """

    raw = invoke_llm(prompt).strip()
    decision = "approve" if "approve" in raw.lower() else "revise"

    reason_line = next((line for line in raw.splitlines() if line.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No Reason"

    state['reasoning_decision'] = decision
    state['reasoning_feedback'] = reason

    write_text_file("reasoning_eval.txt", raw)
    return state

# ----------------Output Reviewer----------------
def reviewer_agent(state):
    state['reviewer_calls']+=1
    
    prompt = f"""
    Evaluate final answer quality.

    Query: {state['user_query']}

    Answer: {state['draft_response']}

    Return EXACTLY:
    Decision: approve OR revise
    Reason: <short reason>
    """

    raw = invoke_llm(prompt).strip()
    decision = "approve" if "approve" in raw.lower() else "revise"

    reason_line = next((line for line in raw.splitlines() if line.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No Reason"

    state['review_decision'] = decision
    state['review_reason'] = reason

    write_text_file(f"reviewer_output_{state['reviewer_calls']}.txt", raw)
    return state

# ----------------Router----------------
def review_router(state):
    if (
        state.get('review_decision') == "approve" 
        and state.get('reasoning_decision') == "approve"
        and state.get('behavior_decision') == "approve"
    ) or state.get('review_count', 0) >= 2:
        return END
    
    state['review_count'] += 1
    return "worker"

# --------------Graph------------------
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
    review_router,
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

result = app.invoke(initial_state)

print("\n=== Final Output ===")
print(result.get("draft_response", ""))
