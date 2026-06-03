"""
Multi-Agent LLM Dashboard
=========================
Streamlit application integrating 4 multi-agent LangGraph workflows powered by Groq.

Pipelines available:
  - Basic        : Planner → Worker → Reviewer (iterative)
  - Reasoning    : + Reasoning Evaluator (logical trace scoring)
  - Behaviour    : + Behaviour Evaluator (tool-usage scoring)
  - System       : + System Metrics (execution time, efficiency score)

━━━ Local Setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install streamlit langgraph langchain-groq python-dotenv plotly
  echo "GROQ_API_KEY=your_key_here" > .env
  streamlit run streamlit_app.py

━━━ Deploy to Streamlit Cloud (free public URL) ━━━━━━━━━━━━━━
  1. Push this file + requirements.txt to a public GitHub repo
  2. Visit https://share.streamlit.io → "New app"
  3. Connect your GitHub repo, set main file = streamlit_app.py
  4. In Settings → Secrets, add:
       GROQ_API_KEY = "your_key_here"
  5. Click Deploy → get shareable URL

requirements.txt contents:
  streamlit>=1.35.0
  langgraph>=0.1.0
  langchain-groq>=0.1.0
  python-dotenv>=1.0.0
  plotly>=5.20.0
  groq>=0.9.0
"""

import os
import time
import logging
from io import StringIO
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Multi-Agent LLM Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root theme */
    :root {
        --bg: #0d0f14;
        --surface: #161920;
        --surface2: #1e2130;
        --border: #2a2f42;
        --accent: #5b7cfa;
        --accent2: #f97d4a;
        --success: #3ecf8e;
        --warn: #f5c542;
        --text: #e4e8f4;
        --muted: #7a82a0;
        --mono: 'Space Mono', monospace;
        --sans: 'DM Sans', sans-serif;
    }

    /* Global resets */
    .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--sans); }
    .stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border); }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d0f14 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(91,124,250,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: var(--mono);
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 6px;
        letter-spacing: -0.5px;
    }
    .hero-title span { color: var(--accent); }
    .hero-sub {
        font-size: 0.92rem;
        color: var(--muted);
        font-weight: 300;
        margin: 0;
        font-family: var(--sans);
    }

    /* Cards */
    .agent-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 14px;
        transition: border-color 0.2s;
    }
    .agent-card:hover { border-color: var(--accent); }
    .agent-card-title {
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .agent-card-body {
        font-size: 0.88rem;
        color: var(--text);
        line-height: 1.7;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Metric tiles */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-tile {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: var(--mono);
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        font-family: var(--mono);
        color: var(--accent);
    }
    .metric-value.good { color: var(--success); }
    .metric-value.warn { color: var(--warn); }
    .metric-value.bad  { color: #f06a6a; }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-family: var(--mono);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-approve { background: rgba(62,207,142,0.15); color: var(--success); border: 1px solid rgba(62,207,142,0.3); }
    .badge-revise  { background: rgba(245,197,66,0.15);  color: var(--warn);    border: 1px solid rgba(245,197,66,0.3); }

    /* Pipeline flow */
    .pipeline-flow {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        padding: 14px 0;
    }
    .pipeline-node {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.78rem;
        font-family: var(--mono);
        color: var(--text);
    }
    .pipeline-node.active { border-color: var(--accent); color: var(--accent); }
    .pipeline-arrow { color: var(--muted); font-size: 1rem; }

    /* Log block */
    .log-block {
        background: #0a0c10;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        font-family: var(--mono);
        font-size: 0.78rem;
        color: #8fa3b8;
        max-height: 320px;
        overflow-y: auto;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    /* Action trace table */
    .action-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    .action-table th {
        background: var(--surface2);
        color: var(--accent);
        font-family: var(--mono);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }
    .action-table td {
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
        color: var(--text);
        vertical-align: top;
        word-break: break-word;
    }
    .action-table tr:last-child td { border-bottom: none; }
    .action-table tr:hover td { background: var(--surface2); }

    /* Sidebar elements */
    .sidebar-section {
        font-family: var(--mono);
        font-size: 0.72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 18px 0 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }

    /* Streamlit overrides */
    .stButton > button {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: var(--mono) !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px !important;
        padding: 10px 28px !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    .stTextInput input, .stTextArea textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: var(--sans) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(91,124,250,0.12) !important;
    }
    .stSelectbox > div > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: #000000 !important;
    }

    .stSelectbox > div > div * {
    color: #000000 !important;
    }

    /* Dropdown menu items */
    li[role="option"] {
    color: #000000 !important;
    }

    /* Selected option */
    li[aria-selected="true"] {
    color: #000000 !important;
    }

    /* BaseWeb Select text */
    div[data-baseweb="select"] span {
    color: #000000 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--mono) !important;
        font-size: 0.78rem !important;
        color: var(--muted) !important;
        border-radius: 6px !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #fff !important;
    }
    div[data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .stAlert {
        border-radius: 8px !important;
    }
    h1,h2,h3,h4,h5 { font-family: var(--mono) !important; color: var(--text) !important; }
    p, li { color: var(--text) !important; font-family: var(--sans) !important; }
    label { color: var(--muted) !important; font-family: var(--sans) !important; font-size: 0.88rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Lazy imports (only after API key confirmed) ─────────────────────────────
def _lazy_imports():
    """Import heavy agent deps lazily so the UI loads fast without them."""
    try:
        from langgraph.graph import StateGraph, END
        from langchain_groq import ChatGroq
        import groq as groq_lib
        return StateGraph, END, ChatGroq, groq_lib
    except ImportError as exc:
        st.error(
            f"Missing dependency: `{exc}`. "
            "Run `pip install langgraph langchain-groq groq` then restart."
        )
        st.stop()

# ─── Logging helpers ─────────────────────────────────────────────────────────
def _make_log_buffer() -> tuple[logging.Logger, StringIO]:
    """Return (logger, StringIO buffer) capturing all log output in memory."""
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    log = logging.getLogger(f"dashboard_{id(buf)}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(handler)
    log.propagate = False
    return log, buf

# ─── Shared LLM factory ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build_llm(api_key: str, model: str):
    """Cache LLM instance per (api_key, model) combo."""
    os.environ["GROQ_API_KEY"] = api_key
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, groq_api_key=api_key)

def _invoke(llm, prompt: str) -> str:
    """Invoke LLM; surface clean error messages on rate-limit / auth failures."""
    try:
        import groq as groq_lib
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        msg = str(exc)
        if "rate_limit" in msg.lower() or "429" in msg:
            raise RuntimeError("Groq rate limit hit. Wait ~30 s or switch model.") from exc
        if "auth" in msg.lower() or "401" in msg:
            raise RuntimeError("Invalid Groq API key. Check sidebar settings.") from exc
        raise

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DEFINITIONS
#  Each pipeline returns a rich state dict ready for the dashboard to render.
# ══════════════════════════════════════════════════════════════════════════════

# ─── Dummy tools (mock external services) ────────────────────────────────────
def search_tool(q):  return f"[search] results for: {q}"
def code_tool(t):    return f"[code] executed: {t}"
def db_tool(q):      return f"[db] retrieved data for: {q}"

def _dispatch_tool(action_name: str, action_input: str) -> str:
    """Route action name → tool function."""
    n = action_name.lower()
    if "search" in n:  return search_tool(action_input)
    if "code"   in n:  return code_tool(action_input)
    if "db"     in n:  return db_tool(action_input)
    return f"[unknown tool: {action_name}]"


# ─── 1. BASIC PIPELINE (app.py) ───────────────────────────────────────────────
def run_basic_pipeline(query: str, llm) -> dict:
    """
    Planner → Worker → Reviewer (loop ≤ 2 revisions).
    Maps to original app.py logic.
    """
    StateGraph_, END_, *_ = _lazy_imports()
    log, buf = _make_log_buffer()
    traces = {"planner": "", "workers": [], "reviewers": []}

    def planner(state):
        log.info("Planner started")
        prompt = (
            f"You are a planning agent. Read the user's query and create "
            f"a short actionable plan for the worker agent.\nUser Query: {state['query']}\n"
            f"Return only a short actionable plan."
        )
        state["plan"] = _invoke(llm, prompt)
        traces["planner"] = state["plan"]
        log.info("Planner done")
        return state

    def worker(state):
        state["worker_calls"] += 1
        fb = state.get("review_reason", "")
        log.info(f"Worker call #{state['worker_calls']}")
        prompt = (
            f"You are a worker agent.\nUser Query: {state['query']}\n"
            f"Plan: {state['plan']}\nPrevious reviewer feedback: {fb}\n\n"
            f"Write the best possible response. If feedback exists, address it."
        )
        state["draft"] = _invoke(llm, prompt)
        traces["workers"].append(state["draft"])
        log.info(f"Worker #{state['worker_calls']} done")
        return state

    def reviewer(state):
        state["reviewer_calls"] += 1
        log.info(f"Reviewer call #{state['reviewer_calls']}")
        prompt = (
            f"You are a strict reviewer. Evaluate this draft response.\n"
            f"User Query: {state['query']}\nDraft: {state['draft']}\n\n"
            f"Check: concrete examples, implementation details, trade-offs, clarity, actionability.\n"
            f"Return EXACTLY:\nDecision: approve OR revise\nReason: brief reason"
        )
        raw = _invoke(llm, prompt).strip()
        state["review_decision"] = "approve" if "decision: approve" in raw.lower() else "revise"
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        state["review_reason"] = reason_line.replace("Reason:", "").strip() or "No reason provided"
        traces["reviewers"].append({
            "call": state["reviewer_calls"],
            "decision": state["review_decision"],
            "reason": state["review_reason"],
        })
        log.info(f"Reviewer #{state['reviewer_calls']}: {state['review_decision']}")
        return state

    def router(state):
        if state.get("review_decision") == "approve" or state.get("review_count", 0) >= 2:
            return END_
        state["review_count"] = state.get("review_count", 0) + 1
        return "worker"

    wf = StateGraph_(dict)
    wf.add_node("planner", planner)
    wf.add_node("worker", worker)
    wf.add_node("reviewer", reviewer)
    wf.set_entry_point("planner")
    wf.add_edge("planner", "worker")
    wf.add_edge("worker", "reviewer")
    wf.add_conditional_edges("reviewer", router, {"worker": "worker", END_: END_})
    app = wf.compile()

    init = {
        "query": query, "plan": "", "draft": "",
        "review_reason": "", "review_decision": "",
        "worker_calls": 0, "reviewer_calls": 0, "review_count": 0,
    }
    result = app.invoke(init)
    return {
        "final_answer": result.get("draft", ""),
        "plan": traces["planner"],
        "worker_drafts": traces["workers"],
        "reviewer_calls": traces["reviewers"],
        "logs": buf.getvalue(),
        "mode": "basic",
    }


# ─── 2. REASONING PIPELINE (reasoning_eval.py) ───────────────────────────────
def run_reasoning_pipeline(query: str, llm) -> dict:
    """
    Planner → Worker (with reasoning trace) → Reasoning Eval → Reviewer.
    Maps to reasoning_eval.py logic.
    """
    StateGraph_, END_, *_ = _lazy_imports()
    log, buf = _make_log_buffer()
    traces = {"planner": "", "workers": [], "reviewers": [], "reasoning_evals": []}

    def planner(state):
        log.info("Planner started")
        prompt = (
            f"You are a planning agent. Create a short actionable plan.\n"
            f"User Query: {state['query']}"
        )
        state["plan"] = _invoke(llm, prompt)
        traces["planner"] = state["plan"]
        return state

    def worker(state):
        state["worker_calls"] += 1
        fb = (state.get("review_reason", "") + "\n" + state.get("reason_feedback", "")).strip()
        log.info(f"Worker call #{state['worker_calls']}")
        prompt = (
            f"You are a worker agent. Solve the task step-by-step.\n"
            f"Return EXACTLY:\n\nFINAL ANSWER:\n<answer>\n\nREASONING TRACE:\n- Step 1:\n- Step 2:\n- Step 3:\n\n"
            f"User Query: {state['query']}\nPlan: {state['plan']}\nFeedback: {fb}"
        )
        output = _invoke(llm, prompt)
        if "REASONING TRACE:" in output:
            parts = output.split("REASONING TRACE:")
            state["draft"] = parts[0].replace("FINAL ANSWER:", "").strip()
            state["reasoning_trace"] = parts[1].strip()
        else:
            state["draft"] = output
            state["reasoning_trace"] = ""
        traces["workers"].append({"draft": state["draft"], "trace": state["reasoning_trace"]})
        log.info(f"Worker #{state['worker_calls']} done")
        return state

    def reasoning_eval(state):
        log.info("Reasoning evaluator started")
        prompt = (
            f"Evaluate this reasoning trace for logical consistency, missing steps, "
            f"incorrect assumptions, hallucinations, and coherence.\n\n"
            f"User Query: {state['query']}\nReasoning Trace:\n{state.get('reasoning_trace', '')}\n\n"
            f"Return EXACTLY:\nScore: <0-10>\nDecision: approve OR revise\nReason: <short explanation>"
        )
        raw = _invoke(llm, prompt).strip()
        state["reasoning_decision"] = "approve" if "approve" in raw.lower() else "revise"
        score_line  = next((l for l in raw.splitlines() if l.lower().startswith("score:")), "")
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        score  = score_line.replace("Score:", "").strip()  or "N/A"
        reason = reason_line.replace("Reason:", "").strip() or "No reason"
        state["reasoning_score"]  = score
        state["reason_feedback"]  = reason
        traces["reasoning_evals"].append({"score": score, "decision": state["reasoning_decision"], "reason": reason})
        log.info(f"Reasoning eval: {state['reasoning_decision']} (score={score})")
        return state

    def reviewer(state):
        state["reviewer_calls"] += 1
        log.info(f"Reviewer call #{state['reviewer_calls']}")
        prompt = (
            f"You are a strict reviewer. Evaluate the final answer for examples, "
            f"implementation details, clarity, and actionability.\n"
            f"User Query: {state['query']}\nAnswer: {state['draft']}\n\n"
            f"Return EXACTLY:\nDecision: approve OR revise\nReason: <short reason>"
        )
        raw = _invoke(llm, prompt).strip()
        state["review_decision"] = "approve" if "approve" in raw.lower() else "revise"
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        state["review_reason"] = reason_line.replace("Reason:", "").strip() or "No reason"
        traces["reviewers"].append({
            "call": state["reviewer_calls"],
            "decision": state["review_decision"],
            "reason": state["review_reason"],
        })
        log.info(f"Reviewer #{state['reviewer_calls']}: {state['review_decision']}")
        return state

    def router(state):
        if (
            state.get("review_decision") == "approve"
            and state.get("reasoning_decision") == "approve"
        ) or state.get("revision_count", 0) >= 2:
            return END_
        state["revision_count"] = state.get("revision_count", 0) + 1
        return "worker"

    wf = StateGraph_(dict)
    wf.add_node("planner", planner)
    wf.add_node("worker", worker)
    wf.add_node("reasoning_eval", reasoning_eval)
    wf.add_node("reviewer", reviewer)
    wf.set_entry_point("planner")
    wf.add_edge("planner", "worker")
    wf.add_edge("worker", "reasoning_eval")
    wf.add_edge("reasoning_eval", "reviewer")
    wf.add_conditional_edges("reviewer", router, {"worker": "worker", END_: END_})
    app = wf.compile()

    init = {
        "query": query, "plan": "", "draft": "", "reasoning_trace": "",
        "review_reason": "", "review_decision": "",
        "reasoning_decision": "", "reason_feedback": "",
        "worker_calls": 0, "reviewer_calls": 0, "revision_count": 0,
    }
    result = app.invoke(init)
    return {
        "final_answer": result.get("draft", ""),
        "plan": traces["planner"],
        "worker_drafts": [w["draft"] for w in traces["workers"]],
        "reasoning_traces": [w["trace"] for w in traces["workers"]],
        "reviewer_calls": traces["reviewers"],
        "reasoning_evals": traces["reasoning_evals"],
        "logs": buf.getvalue(),
        "mode": "reasoning",
    }


# ─── 3. BEHAVIOUR PIPELINE (behaviour_eval.py) ───────────────────────────────
def run_behaviour_pipeline(query: str, llm) -> dict:
    """
    Planner → Worker (tool-use ReAct) → Behaviour Eval → Reasoning Eval → Reviewer.
    Maps to behaviour_eval.py logic.
    """
    StateGraph_, END_, *_ = _lazy_imports()
    log, buf = _make_log_buffer()
    traces = {
        "planner": "", "workers": [], "reviewers": [],
        "behaviour_evals": [], "reasoning_evals": [], "action_traces": [],
    }

    def planner(state):
        log.info("Planner started")
        prompt = (
            f"You are a planning agent. Create a short actionable plan.\n"
            f"User Query: {state['query']}"
        )
        state["plan"] = _invoke(llm, prompt)
        traces["planner"] = state["plan"]
        return state

    def worker(state):
        state["worker_calls"] += 1
        fb = (
            state.get("behavior_feedback", "") + "\n" +
            state.get("reasoning_feedback", "") + "\n" +
            state.get("review_reason", "")
        ).strip()
        log.info(f"Worker call #{state['worker_calls']}")
        prompt = (
            f"You are an AI agent with tools:\n"
            f"- search_tool: factual web searches\n"
            f"- code_tool: computations and code execution\n"
            f"- db_tool: structured database queries\n\n"
            f"RULES: Choose correct tool. One tool per step. Do not hallucinate.\n\n"
            f"Format:\nThought:\nAction:\nAction Input:\n(repeat if needed)\n\n"
            f"Then:\nFINAL ANSWER:\n<answer>\n\nREASONING TRACE:\n- Step 1:\n- Step 2:\n\n"
            f"User Query: {state['query']}\nPlan: {state['plan']}\nFeedback: {fb}"
        )
        output = _invoke(llm, prompt)
        lines = output.splitlines()
        final_answer = ""
        reasoning_trace = ""
        action_trace = []
        current_action = None

        for i, line in enumerate(lines):
            ll = line.lower()
            if ll.startswith("action:"):
                current_action = {"action": line.split(":", 1)[1].strip()}
            elif ll.startswith("action input:") and current_action:
                inp = line.split(":", 1)[1].strip()
                result = _dispatch_tool(current_action["action"], inp)
                action_trace.append({"action": current_action["action"], "input": inp, "result": result})
                current_action = None
            elif ll.startswith("final answer:"):
                final_answer = line.split(":", 1)[1].strip()
            elif ll.startswith("reasoning trace:"):
                reasoning_trace = "\n".join(lines[i + 1:]).strip()

        state["draft"] = final_answer or output
        state["reasoning_trace"] = reasoning_trace
        state["action_trace"] = action_trace
        traces["workers"].append(state["draft"])
        traces["action_traces"].append(action_trace)
        log.info(f"Worker #{state['worker_calls']} done — {len(action_trace)} tool calls")
        return state

    def behaviour_eval(state):
        log.info("Behaviour evaluator started")
        prompt = (
            f"You are a behaviour evaluator. Score tool usage quality.\n"
            f"Tools: search_tool (factual), code_tool (computational), db_tool (structured).\n\n"
            f"User Query: {state['query']}\nAction Trace: {state.get('action_trace', [])}\n\n"
            f"Check: correct tool, no hallucination, efficiency, logical order.\n"
            f"Return EXACTLY:\nScore: <0-10>\nDecision: approve OR revise\nReason: <short explanation>"
        )
        raw = _invoke(llm, prompt).strip()
        state["behavior_decision"] = "approve" if "approve" in raw.lower() else "revise"
        score_line  = next((l for l in raw.splitlines() if l.lower().startswith("score:")), "")
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        score  = score_line.replace("Score:", "").strip()  or "N/A"
        reason = reason_line.replace("Reason:", "").strip() or "No reason"
        state["behavior_feedback"] = reason
        traces["behaviour_evals"].append({"score": score, "decision": state["behavior_decision"], "reason": reason})
        log.info(f"Behaviour eval: {state['behavior_decision']} (score={score})")
        return state

    def reasoning_eval(state):
        log.info("Reasoning evaluator started")
        prompt = (
            f"Evaluate reasoning trace quality.\n"
            f"User Query: {state['query']}\nReasoning Trace: {state.get('reasoning_trace', '')}\n\n"
            f"Return EXACTLY:\nScore: <0-10>\nDecision: approve OR revise\nReason: <short explanation>"
        )
        raw = _invoke(llm, prompt).strip()
        state["reasoning_decision"] = "approve" if "approve" in raw.lower() else "revise"
        score_line  = next((l for l in raw.splitlines() if l.lower().startswith("score:")), "")
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        score  = score_line.replace("Score:", "").strip()  or "N/A"
        reason = reason_line.replace("Reason:", "").strip() or "No reason"
        state["reasoning_feedback"] = reason
        traces["reasoning_evals"].append({"score": score, "decision": state["reasoning_decision"], "reason": reason})
        log.info(f"Reasoning eval: {state['reasoning_decision']} (score={score})")
        return state

    def reviewer(state):
        state["reviewer_calls"] += 1
        log.info(f"Reviewer call #{state['reviewer_calls']}")
        prompt = (
            f"Evaluate final answer quality.\nUser Query: {state['query']}\n"
            f"Answer: {state['draft']}\n\n"
            f"Return EXACTLY:\nDecision: approve OR revise\nReason: <short reason>"
        )
        raw = _invoke(llm, prompt).strip()
        state["review_decision"] = "approve" if "approve" in raw.lower() else "revise"
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        state["review_reason"] = reason_line.replace("Reason:", "").strip() or "No reason"
        traces["reviewers"].append({
            "call": state["reviewer_calls"],
            "decision": state["review_decision"],
            "reason": state["review_reason"],
        })
        log.info(f"Reviewer #{state['reviewer_calls']}: {state['review_decision']}")
        return state

    def router(state):
        if (
            state.get("review_decision") == "approve"
            and state.get("reasoning_decision") == "approve"
            and state.get("behavior_decision") == "approve"
        ) or state.get("revision_count", 0) >= 2:
            return END_
        state["revision_count"] = state.get("revision_count", 0) + 1
        return "worker"

    wf = StateGraph_(dict)
    wf.add_node("planner", planner)
    wf.add_node("worker", worker)
    wf.add_node("behaviour_eval", behaviour_eval)
    wf.add_node("reasoning_eval", reasoning_eval)
    wf.add_node("reviewer", reviewer)
    wf.set_entry_point("planner")
    wf.add_edge("planner", "worker")
    wf.add_edge("worker", "behaviour_eval")
    wf.add_edge("behaviour_eval", "reasoning_eval")
    wf.add_edge("reasoning_eval", "reviewer")
    wf.add_conditional_edges("reviewer", router, {"worker": "worker", END_: END_})
    app = wf.compile()

    init = {
        "query": query, "plan": "", "draft": "",
        "reasoning_trace": "", "action_trace": [],
        "review_reason": "", "review_decision": "",
        "reasoning_decision": "", "reasoning_feedback": "",
        "behavior_decision": "", "behavior_feedback": "",
        "worker_calls": 0, "reviewer_calls": 0, "revision_count": 0,
    }
    result = app.invoke(init)
    return {
        "final_answer": result.get("draft", ""),
        "plan": traces["planner"],
        "worker_drafts": traces["workers"],
        "reviewer_calls": traces["reviewers"],
        "behaviour_evals": traces["behaviour_evals"],
        "reasoning_evals": traces["reasoning_evals"],
        "action_traces": traces["action_traces"],
        "logs": buf.getvalue(),
        "mode": "behaviour",
    }


# ─── 4. SYSTEM PIPELINE (system_eval.py) ─────────────────────────────────────
def run_system_pipeline(query: str, llm) -> dict:
    """
    Full pipeline + system performance metrics (execution time, revision count, tool calls).
    Maps to system_eval.py logic.
    """
    StateGraph_, END_, *_ = _lazy_imports()
    log, buf = _make_log_buffer()
    traces = {
        "planner": "", "workers": [], "reviewers": [],
        "behaviour_evals": [], "reasoning_evals": [], "action_traces": [],
    }

    def planner(state):
        log.info("Planner started")
        state["plan"] = _invoke(llm, f"Create a short plan for:\n{state['query']}")
        traces["planner"] = state["plan"]
        return state

    def worker(state):
        state["worker_calls"] += 1
        fb = (
            state.get("behavior_feedback", "") + "\n" +
            state.get("reasoning_feedback", "") + "\n" +
            state.get("review_reason", "")
        ).strip()
        log.info(f"Worker call #{state['worker_calls']}")
        prompt = (
            f"You are an AI agent with tools:\n"
            f"  search_tool → facts\n  code_tool → computation\n  db_tool → structured data\n\n"
            f"RULES: choose correct tool, one tool per step.\n\n"
            f"Format:\nThought:\nAction:\nAction Input:\n(repeat)\n\nFinal Answer:\n<answer>\n\n"
            f"User Query: {state['query']}\nPlan: {state['plan']}\nFeedback: {fb}"
        )
        output = _invoke(llm, prompt)
        lines = output.splitlines()
        final_answer = ""
        action_trace = []
        current_action = None

        for line in lines:
            ll = line.lower()
            if ll.startswith("action:"):
                current_action = {"action": line.split(":", 1)[1].strip()}
            elif ll.startswith("action input:") and current_action:
                inp = line.split(":", 1)[1].strip()
                result = _dispatch_tool(current_action["action"], inp)
                action_trace.append({"action": current_action["action"], "input": inp, "result": result})
                current_action = None
            elif "final answer:" in ll:
                final_answer = line.split(":", 1)[1].strip()

        state["draft"] = final_answer or output
        state["action_trace"] = action_trace
        traces["workers"].append(state["draft"])
        traces["action_traces"].append(action_trace)
        log.info(f"Worker #{state['worker_calls']} done")
        return state

    def behaviour_eval(state):
        log.info("Behaviour evaluator started")
        prompt = (
            f"Evaluate tool usage:\nQuery: {state['query']}\n"
            f"Action Trace: {state.get('action_trace', [])}\n\n"
            f"Score correct tool usage, efficiency, correctness.\n"
            f"Return:\nScore: 0-10\nDecision: approve OR revise\nReason: short"
        )
        raw = _invoke(llm, prompt)
        state["behavior_decision"] = "approve" if "approve" in raw.lower() else "revise"
        score_line  = next((l for l in raw.splitlines() if l.lower().startswith("score:")), "")
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        state["behavior_feedback"] = reason_line.replace("Reason:", "").strip() or raw
        score = score_line.replace("Score:", "").strip() or "N/A"
        traces["behaviour_evals"].append({"score": score, "decision": state["behavior_decision"], "reason": state["behavior_feedback"]})
        log.info(f"Behaviour eval: {state['behavior_decision']}")
        return state

    def reasoning_eval(state):
        log.info("Reasoning evaluator started")
        prompt = (
            f"Evaluate reasoning quality:\nQuery: {state['query']}\n\n"
            f"Return:\nScore: 0-10\nDecision: approve OR revise\nReason: short"
        )
        raw = _invoke(llm, prompt)
        state["reasoning_decision"] = "approve" if "approve" in raw.lower() else "revise"
        score_line  = next((l for l in raw.splitlines() if l.lower().startswith("score:")), "")
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        state["reasoning_feedback"] = reason_line.replace("Reason:", "").strip() or raw
        score = score_line.replace("Score:", "").strip() or "N/A"
        traces["reasoning_evals"].append({"score": score, "decision": state["reasoning_decision"], "reason": state["reasoning_feedback"]})
        log.info(f"Reasoning eval: {state['reasoning_decision']}")
        return state

    def reviewer(state):
        state["reviewer_calls"] += 1
        log.info(f"Reviewer call #{state['reviewer_calls']}")
        prompt = (
            f"Evaluate final answer:\n{state['draft']}\n\n"
            f"Return:\nScore: 0-10\nDecision: approve OR revise\nReason: short"
        )
        raw = _invoke(llm, prompt)
        state["review_decision"] = "approve" if "approve" in raw.lower() else "revise"
        reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
        state["review_reason"] = reason_line.replace("Reason:", "").strip() or raw
        traces["reviewers"].append({
            "call": state["reviewer_calls"],
            "decision": state["review_decision"],
            "reason": state["review_reason"],
        })
        log.info(f"Reviewer #{state['reviewer_calls']}: {state['review_decision']}")
        return state

    def router(state):
        # Bug fix from original: reviewer_decision key → review_decision
        if (
            state.get("review_decision") == "approve"
            and state.get("reasoning_decision") == "approve"
            and state.get("behavior_decision") == "approve"
        ) or state.get("revision_count", 0) >= 2:
            return "__end__"
        state["revision_count"] = state.get("revision_count", 0) + 1
        return "worker"

    wf = StateGraph_(dict)
    wf.add_node("planner", planner)
    wf.add_node("worker", worker)
    wf.add_node("behaviour_eval", behaviour_eval)
    wf.add_node("reasoning_eval", reasoning_eval)
    wf.add_node("reviewer", reviewer)
    wf.set_entry_point("planner")
    wf.add_edge("planner", "worker")
    wf.add_edge("worker", "behaviour_eval")
    wf.add_edge("behaviour_eval", "reasoning_eval")
    wf.add_edge("reasoning_eval", "reviewer")
    wf.add_conditional_edges("reviewer", router, {"worker": "worker", "__end__": END_})
    app = wf.compile()

    init = {
        "query": query, "plan": "", "draft": "",
        "reasoning_trace": "", "action_trace": [],
        "review_reason": "", "review_decision": "",
        "reasoning_decision": "", "reasoning_feedback": "",
        "behavior_decision": "", "behavior_feedback": "",
        "worker_calls": 0, "reviewer_calls": 0, "revision_count": 0,
    }
    t0 = time.time()
    result = app.invoke(init)
    elapsed = time.time() - t0

    # System score (adapted from system_eval.py, with total_calls fix)
    worker_c   = result.get("worker_calls", 0)
    reviewer_c = result.get("reviewer_calls", 0)
    revision_c = result.get("revision_count", 0)
    tool_c     = sum(len(t) for t in traces["action_traces"])
    total_c    = worker_c + reviewer_c

    score = 10
    if elapsed    > 10: score -= 2
    if revision_c > 2:  score -= 2
    if total_c    > 6:  score -= 2
    if worker_c   > 3:  score -= 2
    score = max(score, 0)

    metrics = {
        "execution_time": round(elapsed, 2),
        "worker_calls":   worker_c,
        "reviewer_calls": reviewer_c,
        "revision_count": revision_c,
        "tool_calls":     tool_c,
        "total_calls":    total_c,
        "system_score":   score,
    }

    return {
        "final_answer":    result.get("draft", ""),
        "plan":            traces["planner"],
        "worker_drafts":   traces["workers"],
        "reviewer_calls":  traces["reviewers"],
        "behaviour_evals": traces["behaviour_evals"],
        "reasoning_evals": traces["reasoning_evals"],
        "action_traces":   traces["action_traces"],
        "metrics":         metrics,
        "logs":            buf.getvalue(),
        "mode":            "system",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  UI RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _badge(decision: str) -> str:
    cls = "badge-approve" if decision == "approve" else "badge-revise"
    return f'<span class="badge {cls}">{decision}</span>'

def _pipeline_nodes(mode: str) -> list[str]:
    base = ["Planner", "Worker", "Reviewer"]
    if mode == "reasoning": return ["Planner", "Worker", "Reasoning Eval", "Reviewer"]
    if mode in ("behaviour", "system"): return ["Planner", "Worker", "Behaviour Eval", "Reasoning Eval", "Reviewer"]
    return base

def render_pipeline_diagram(mode: str):
    nodes = _pipeline_nodes(mode)
    html = '<div class="pipeline-flow">'
    for i, node in enumerate(nodes):
        html += f'<div class="pipeline-node active">{node}</div>'
        if i < len(nodes) - 1:
            html += '<span class="pipeline-arrow">→</span>'
    html += '<span class="pipeline-arrow">⟳</span><span style="font-size:0.75rem;color:var(--muted);font-family:var(--mono)">loop ≤ 2</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_agent_card(title: str, content: str):
    safe = content.replace("<", "&lt;").replace(">", "&gt;") if content else "<em style='color:var(--muted)'>— empty —</em>"
    st.markdown(
        f'<div class="agent-card">'
        f'<div class="agent-card-title">{title}</div>'
        f'<div class="agent-card-body">{safe}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_results(data: dict):
    """Main results renderer — tabs vary by pipeline mode."""
    mode = data.get("mode", "basic")

    # Build tab list based on mode
    tab_labels = ["📝 Final Answer", "🗂 Agent Traces"]
    if mode in ("reasoning", "behaviour", "system"):
        tab_labels.append("📊 Evaluations")
    if mode == "system":
        tab_labels.append("📈 System Metrics")
    tab_labels.append("📋 Logs")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ── Tab: Final Answer ──────────────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("### Final Answer")
        answer = data.get("final_answer", "")
        if answer:
            st.markdown(answer)
        else:
            st.warning("No final answer returned.")

        if data.get("plan"):
            with st.expander("🗺 Planner Output"):
                st.markdown(data["plan"])

    # ── Tab: Agent Traces ──────────────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        worker_drafts = data.get("worker_drafts", [])
        reviewer_calls = data.get("reviewer_calls", [])
        action_traces = data.get("action_traces", [])
        reasoning_traces = data.get("reasoning_traces", [])

        if not worker_drafts and not reviewer_calls:
            st.info("No trace data available.")
        else:
            for i, draft in enumerate(worker_drafts):
                render_agent_card(f"Worker Draft #{i + 1}", draft)

                # Show action trace if available (behaviour/system modes)
                if action_traces and i < len(action_traces) and action_traces[i]:
                    st.markdown(
                        '<div class="agent-card">'
                        '<div class="agent-card-title">Tool Calls</div>',
                        unsafe_allow_html=True,
                    )
                    rows = action_traces[i]
                    html = '<table class="action-table"><thead><tr><th>#</th><th>Action</th><th>Input</th><th>Result</th></tr></thead><tbody>'
                    for j, row in enumerate(rows):
                        html += (
                            f"<tr><td>{j+1}</td>"
                            f"<td><code>{row.get('action','')}</code></td>"
                            f"<td>{row.get('input','')[:80]}</td>"
                            f"<td>{row.get('result','')[:100]}</td></tr>"
                        )
                    html += "</tbody></table></div>"
                    st.markdown(html, unsafe_allow_html=True)

                # Show reasoning trace if available (reasoning mode)
                if reasoning_traces and i < len(reasoning_traces) and reasoning_traces[i]:
                    with st.expander(f"🧠 Reasoning Trace #{i + 1}"):
                        st.text(reasoning_traces[i])

            st.markdown("---")
            for rv in reviewer_calls:
                st.markdown(
                    f"**Reviewer Call #{rv['call']}** — "
                    + _badge(rv["decision"])
                    + f"  &nbsp; *{rv.get('reason', '')}*",
                    unsafe_allow_html=True,
                )

    # ── Tab: Evaluations ───────────────────────────────────────────────────
    if mode in ("reasoning", "behaviour", "system"):
        with tabs[tab_idx]:
            tab_idx += 1
            behaviour_evals = data.get("behaviour_evals", [])
            reasoning_evals = data.get("reasoning_evals", [])

            if behaviour_evals:
                st.markdown("#### 🛠 Behaviour Evaluations")
                for i, ev in enumerate(behaviour_evals):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        score_val = ev.get("score", "N/A")
                        try:
                            num = float(str(score_val).split("/")[0])
                            color = "good" if num >= 7 else ("warn" if num >= 4 else "bad")
                        except ValueError:
                            color = "warn"
                        st.markdown(
                            f'<div class="metric-tile">'
                            f'<div class="metric-label">Score #{i+1}</div>'
                            f'<div class="metric-value {color}">{score_val}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            _badge(ev["decision"]) + f"  &nbsp; {ev.get('reason', '')}",
                            unsafe_allow_html=True,
                        )

            if reasoning_evals:
                st.markdown("#### 🧠 Reasoning Evaluations")
                for i, ev in enumerate(reasoning_evals):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        score_val = ev.get("score", "N/A")
                        try:
                            num = float(str(score_val).split("/")[0])
                            color = "good" if num >= 7 else ("warn" if num >= 4 else "bad")
                        except ValueError:
                            color = "warn"
                        st.markdown(
                            f'<div class="metric-tile">'
                            f'<div class="metric-label">Score #{i+1}</div>'
                            f'<div class="metric-value {color}">{score_val}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            _badge(ev["decision"]) + f"  &nbsp; {ev.get('reason', '')}",
                            unsafe_allow_html=True,
                        )

    # ── Tab: System Metrics ────────────────────────────────────────────────
    if mode == "system":
        with tabs[tab_idx]:
            tab_idx += 1
            metrics = data.get("metrics", {})
            if metrics:
                score = metrics.get("system_score", 0)
                score_color = "good" if score >= 8 else ("warn" if score >= 5 else "bad")

                # Top metric tiles
                st.markdown(
                    f'<div class="metric-grid">'
                    f'<div class="metric-tile">'
                    f'<div class="metric-label">System Score</div>'
                    f'<div class="metric-value {score_color}">{score}/10</div></div>'
                    f'<div class="metric-tile">'
                    f'<div class="metric-label">Exec Time</div>'
                    f'<div class="metric-value">{metrics.get("execution_time", 0):.1f}s</div></div>'
                    f'<div class="metric-tile">'
                    f'<div class="metric-label">Worker Calls</div>'
                    f'<div class="metric-value">{metrics.get("worker_calls", 0)}</div></div>'
                    f'<div class="metric-tile">'
                    f'<div class="metric-label">Reviewer Calls</div>'
                    f'<div class="metric-value">{metrics.get("reviewer_calls", 0)}</div></div>'
                    f'<div class="metric-tile">'
                    f'<div class="metric-label">Revisions</div>'
                    f'<div class="metric-value">{metrics.get("revision_count", 0)}</div></div>'
                    f'<div class="metric-tile">'
                    f'<div class="metric-label">Tool Calls</div>'
                    f'<div class="metric-value">{metrics.get("tool_calls", 0)}</div></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Bar chart
                try:
                    import plotly.graph_objects as go

                    labels = ["Worker Calls", "Reviewer Calls", "Revisions", "Tool Calls"]
                    values = [
                        metrics.get("worker_calls", 0),
                        metrics.get("reviewer_calls", 0),
                        metrics.get("revision_count", 0),
                        metrics.get("tool_calls", 0),
                    ]
                    fig = go.Figure(
                        go.Bar(
                            x=labels,
                            y=values,
                            marker_color=["#5b7cfa", "#3ecf8e", "#f5c542", "#f97d4a"],
                            text=values,
                            textposition="outside",
                        )
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e4e8f4", family="Space Mono"),
                        xaxis=dict(gridcolor="#2a2f42"),
                        yaxis=dict(gridcolor="#2a2f42"),
                        margin=dict(t=20, b=20),
                        height=280,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.json(metrics)

    # ── Tab: Logs ──────────────────────────────────────────────────────────
    with tabs[tab_idx]:
        logs = data.get("logs", "No logs captured.")
        st.markdown(f'<div class="log-block">{logs}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:1.05rem;'
        'font-weight:700;color:#5b7cfa;padding:8px 0 4px;">⬡ Multi-Agent LLM</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">Configuration</div>', unsafe_allow_html=True)

    # API key — prefer env var, allow override
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    model_options = [
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    model = st.selectbox("Model", model_options, index=0)

    st.markdown('<div class="sidebar-section">Pipeline Mode</div>', unsafe_allow_html=True)

    mode_labels = {
        "🔷 Basic": "basic",
        "🧠 Reasoning Eval": "reasoning",
        "🛠 Behaviour Eval": "behaviour",
        "📈 System Eval": "system",
    }
    selected_label = st.radio("Select pipeline", list(mode_labels.keys()), index=0)
    selected_mode  = mode_labels[selected_label]

    # Mode description
    desc = {
        "basic":     "Planner → Worker → Reviewer loop. Clean and fast.",
        "reasoning": "Adds a Reasoning Evaluator that scores logical trace quality.",
        "behaviour": "Adds Behaviour + Reasoning Evaluators scoring tool usage.",
        "system":    "Full pipeline with execution time & system efficiency scoring.",
    }
    st.caption(desc[selected_mode])

    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.caption(
        "Built with LangGraph · Groq · Streamlit  \n"
        "[Groq Console](https://console.groq.com) · "
        "[Streamlit Cloud](https://share.streamlit.io)"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Hero header
st.markdown(
    '<div class="hero-header">'
    '<div class="hero-title">Multi-Agent <span>LLM</span> Dashboard</div>'
    '<div class="hero-sub">'
    "LangGraph pipelines with Planner → Worker → Evaluators → Reviewer — powered by Groq"
    "</div></div>",
    unsafe_allow_html=True,
)

# Pipeline diagram
render_pipeline_diagram(selected_mode)

st.markdown("---")

# Query input
col_q, col_btn = st.columns([5, 1], gap="medium")
with col_q:
    example_queries = [
        "Explain the trade-offs between REST and GraphQL APIs with implementation examples",
        "How do I implement rate limiting in a FastAPI application?",
        "Compare PostgreSQL vs MongoDB for a social media platform",
        "Write a Python script to scrape and parse structured data from a website",
    ]
    user_query = st.text_area(
        "Your Query",
        height=90,
        placeholder="Ask anything — the agent will plan, research, draft, and refine an answer...",
        label_visibility="collapsed",
    )

with col_btn:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_clicked = st.button("▶ Run", use_container_width=True)

# Quick example chips
st.markdown(
    '<div style="font-size:0.78rem;color:var(--muted);font-family:var(--mono);'
    'margin-bottom:6px;">EXAMPLE QUERIES</div>',
    unsafe_allow_html=True,
)
ex_cols = st.columns(len(example_queries))
for col, ex in zip(ex_cols, example_queries):
    with col:
        if st.button(ex[:45] + "…", key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["prefill"] = ex

# Apply prefill if an example was clicked
if "prefill" in st.session_state and not user_query:
    user_query = st.session_state.pop("prefill")

st.markdown("---")

# ─── Execute Pipeline ────────────────────────────────────────────────────────
if run_clicked:
    if not api_key:
        st.error("Add your Groq API key in the sidebar.")
        st.stop()
    if not user_query.strip():
        st.warning("Enter a query above to continue.")
        st.stop()

    pipeline_fn = {
        "basic":     run_basic_pipeline,
        "reasoning": run_reasoning_pipeline,
        "behaviour": run_behaviour_pipeline,
        "system":    run_system_pipeline,
    }[selected_mode]

    # Progress indicator
    progress_bar = st.progress(0)
    status_text  = st.empty()

    status_messages = [
        ("Initialising agents…",    5),
        ("Planner crafting plan…", 20),
        ("Worker drafting response…", 50),
        ("Evaluators reviewing…",   75),
        ("Finalising answer…",      90),
    ]

    import threading

    run_result: dict[str, Any] = {}
    run_error:  list[str]      = []

    def _run():
        try:
            llm = _build_llm(api_key, model)
            run_result.update(pipeline_fn(user_query.strip(), llm))
        except Exception as exc:
            run_error.append(str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Animate progress while running
    step = 0
    while thread.is_alive():
        if step < len(status_messages):
            msg, pct = status_messages[step]
            status_text.markdown(
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.82rem;'
                f'color:#7a82a0;">{msg}</div>',
                unsafe_allow_html=True,
            )
            progress_bar.progress(pct)
            step += 1
        time.sleep(1.5)

    thread.join()
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    if run_error:
        st.error(f"Pipeline error: {run_error[0]}")
        with st.expander("Debug Info"):
            st.code(run_error[0])
    else:
        st.session_state["last_result"] = run_result
        st.session_state["last_query"]  = user_query

# ─── Render cached results ────────────────────────────────────────────────────
if "last_result" in st.session_state:
    q = st.session_state.get("last_query", "")
    if q:
        st.markdown(
            f'<div style="font-size:0.8rem;color:var(--muted);font-family:var(--mono);'
            f'margin-bottom:12px;">QUERY: <span style="color:var(--text)">{q}</span></div>',
            unsafe_allow_html=True,
        )
    render_results(st.session_state["last_result"])

elif not run_clicked:
    # Empty state
    st.markdown(
        '<div style="text-align:center;padding:60px 0;">'
        '<div style="font-size:3rem;margin-bottom:12px;">🤖</div>'
        '<div style="font-family:\'Space Mono\',monospace;font-size:0.9rem;color:var(--muted);">'
        "Enter a query above and click <strong style='color:var(--accent)'>▶ Run</strong> to start the pipeline."
        "</div></div>",
        unsafe_allow_html=True,
    )
