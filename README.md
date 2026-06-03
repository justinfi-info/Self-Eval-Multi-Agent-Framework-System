# Self-Eval-Multi-Agent-Framework-System

A production-grade, self-evaluating multi-agent AI system built with **Python**, **LangGraph**, and **Groq LLMs**, designed to benchmark agent reasoning, execution quality, critique loops, and self-improvement workflows.

**The Missing Piece in AI Evaluation:** A correct answer doesn't guarantee a correct AI system. This framework teaches you how to evaluate agents at 4 distinct layers, orchestrate complex multi-agent workflows, and build AI systems that automatically improve themselves.

---

## 🎯 The 4-Layer AI Evaluation Framework

Most AI evaluation focuses on **one thing: Is the final answer correct?** But that's incomplete. Real production systems need to answer four critical questions:

```
┌─────────────────────────────────────────────────────────┐
│         Layer 4: System Evaluation                       │
│  Is the system fast, scalable, and cost-effective?      │
├─────────────────────────────────────────────────────────┤
│         Layer 3: Behavior Evaluation                     │
│  Did the agent use tools efficiently?                   │
├─────────────────────────────────────────────────────────┤
│         Layer 2: Reasoning Evaluation                    │
│  Did the model think logically?                         │
├─────────────────────────────────────────────────────────┤
│         Layer 1: Output Evaluation                       │
│  Is the final answer correct?                           │
└─────────────────────────────────────────────────────────┘
```

### Why This Matters

✅ **Layer 1 (Output)** — A chatbot returns a correct answer, but...  
✅ **Layer 2 (Reasoning)** — Its reasoning was circular and illogical, so...  
✅ **Layer 3 (Behavior)** — It called 10 tools instead of 2, wasting API costs, so...  
✅ **Layer 4 (System)** — Your entire system fails in production due to latency and cost overruns.

**You can't fix what you don't measure.** This framework teaches you to measure all four layers simultaneously.

---

## 📊 LangGraph Workflow Architecture

The system orchestrates five specialized agents in a self-critiquing loop:

```
                    ┌──────────────────┐
                    │   USER QUERY     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  PLANNER AGENT   │◄──── Creates initial plan
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  WORKER AGENT    │◄──── Executes with feedback
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Behavior │  │ Reasoning│  │ Output   │
        │  Eval    │  │  Eval    │  │ Reviewer │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             └─────────────┴─────────────┘
                           │
                    ┌──────▼──────┐
                    │ ROUTER      │
                    │ (Decision)  │
                    └──────┬──────┘
                           │
                  ┌────────┴────────┐
                  │                 │
            APPROVED          ITERATE
            (END)             (Loop)
                           (max 2x)
```

### Agent Responsibilities

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **Planner** | Breaks down user query into actionable steps | User query | Strategic plan |
| **Worker** | Executes the plan with tools (search, code, db) | Plan + feedback | Draft response + action trace |
| **Behavior Evaluator** | Scores tool usage, efficiency, correctness | Action trace | Score (0-10) + decision |
| **Reasoning Evaluator** | Analyzes the quality of thinking | Reasoning steps | Score (0-10) + decision |
| **Reviewer** | Evaluates final output for completeness | Draft response | Score (0-10) + decision |

**The Router** decides: Continue to worker for refinement or end the loop?

---

## ✨ Key Features

- **Self-Evaluating Pipeline** — Agents critique each other's work automatically
- **Action Trace Tracking** — Every tool call, every decision, every step is logged
- **Feedback Loops** — Failed evaluations automatically send structured feedback to the worker
- **System-Level Metrics** — Tracks execution time, API calls, iterations, and cost
- **LangGraph Orchestration** — Visual workflow graph, built-in state management, conditional routing
- **Multi-Layer Scoring** — Unified evaluation combining output + reasoning + behavior + system efficiency
- **LLM-as-a-Judge** — Uses LLMs to evaluate other LLM outputs with consistent scoring

---

## 📦 What We Build (End-to-End)

### Part 1 — The Theory: Agentic AI Evaluation Explained

Most people skip the theory. We don't. Before writing a single line of code, we break down why traditional evaluation fails for agents and introduce the 4-layer framework used in real production systems. This mental model will change how you think about AI systems forever.

**Key Concepts:**
- Why accuracy isn't enough
- The fallacy of single-metric evaluation
- How production AI systems actually fail
- The importance of behavioral metrics
- Cost vs. quality trade-offs

### Part 2 — The Build: Multi-Agent System with Self-Critique

We implement a complete **Planner + Worker + Reviewer** pipeline from scratch in Python using **LangGraph** and **Groq's ChatGroq LLM**. The Reviewer Agent checks every response for:
- Clarity and comprehensiveness
- Concrete examples
- Trade-off analysis
- Actionable recommendations

The feedback loop sends critical insights back to the Worker for iterative refinement until approval.

**Files:** `app.py` (simple 3-layer), `system_eval.py` (full 5-agent system)

### Part 3 — Output + Reasoning Evaluation with LLM-as-a-Judge

We extend the system to evaluate its own outputs using dedicated Judge Agents. The **Reasoning Evaluator** analyzes the quality of the agent's thinking process — not just the final result — scoring:
- Logic coherence
- Step-by-step correctness
- Inference validity
- Completeness of analysis

**Files:** `reasoning_eval.py`

### Part 4 — Behavior Evaluation: Does the Agent Know When It Made a Mistake?

This is where it gets powerful. We add a **Behavior Evaluator** that tracks every tool call, every action step, and every decision in a structured action trace. It then scores:
- Were the right tools used?
- Were they used in the optimal order?
- Were there unnecessary steps?
- Was API/compute wasted?

This is how real-world agent evaluation works in production systems like OpenAI's o1, Anthropic's extended thinking, and enterprise AI platforms.

**Files:** `behaviour_eval.py`

### Part 5 — Full Multi-Layer Scoring System

We combine all evaluation layers into a unified scoring engine:

```python
system_score = (output_score × 0.3 +
                reasoning_score × 0.25 +
                behavior_score × 0.25 +
                efficiency_score × 0.2)
```

You'll see how to:
- Weight and combine evaluation signals
- Track system-level efficiency (execution time, API calls, iterations)
- Generate production-grade quality reports
- Identify bottlenecks and optimization opportunities

**Files:** `system_eval.py`, evaluation outputs in `logs/`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Groq API Key (free tier available at [groq.com](https://groq.com))
- LangGraph, LangChain, python-dotenv

### Installation

```bash
# Clone the repository
git clone https://github.com/justinfi-info/Self-Eval-Multi-Agent-Framework-System.git
cd Self-Eval-Multi-Agent-Framework-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Environment
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the System

```bash
# Simple 3-layer system (Planner → Worker → Reviewer)
python app.py

# Full 5-layer system (includes behavior + reasoning evaluation)
python system_eval.py

# Reasoning evaluation specialized system
python reasoning_eval.py

# Behavior evaluation specialized system
python behaviour_eval.py
```

---

## 📁 Project Structure

```
.
├── app.py                          # Simple 3-layer pipeline
├── system_eval.py                  # Full 5-agent multi-layer system
├── reasoning_eval.py               # Reasoning quality evaluation
├── behaviour_eval.py               # Tool usage & behavior analysis
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── logs/                           # Execution traces & outputs
│   ├── action_trace_*.txt          # Tool call sequences
│   ├── worker_output_*.txt         # Draft responses at each iteration
│   ├── reviewer_output_*.txt       # Review feedback
│   ├── behavior_eval.txt           # Behavior evaluation scores
│   ├── reasoning_eval.txt          # Reasoning evaluation scores
│   ├── final_output.txt            # Final approved response
│   └── execution.log               # Full execution trace
└── venv/                           # Python virtual environment
```

---

## 🔧 How It Works: Step-by-Step Execution

### 1. User Submits Query
```
User: "Explain how neural networks work"
```

### 2. Planner Creates Strategic Plan
```
Plan:
  1. Define neural networks briefly
  2. Explain key components: neurons, layers, weights
  3. Describe forward propagation
  4. Explain backpropagation
  5. Provide concrete example (MNIST)
  6. Discuss real-world applications
```

### 3. Worker Executes with Tools
```
Thought: I need to search for current NN architectures
Action: search_tool("neural network architectures 2024")
Result: [Deep Learning papers from 2024...]

Thought: Now I need to write sample code
Action: code_tool("Write simple neural network in PyTorch")
Result: [Working PyTorch code...]

Final Answer: [Comprehensive explanation with examples]
```

### 4. Parallel Evaluation (3 Agents)
```
Behavior Evaluator:
  ✓ Used search tool appropriately
  ✓ Used code tool for concrete example
  ✗ Made 3 API calls (efficient)
  Score: 8/10 → APPROVE

Reasoning Evaluator:
  ✓ Logical flow from simple to complex
  ✓ Explained foundational concepts first
  ✓ Covered both theory and practice
  Score: 9/10 → APPROVE

Reviewer (Output Quality):
  ✓ Clear and comprehensive
  ✓ Has concrete examples
  ✓ Actionable recommendations
  Score: 9/10 → APPROVE
```

### 5. Router Decision
```
All evaluators: APPROVE
System Score: 8.67/10
Status: ✅ COMPLETE (END)
```

---

## 📊 System Metrics & Output

Each execution generates:

- **Action Trace** — Structured log of every tool call
- **Evaluation Scores** — Behavior, reasoning, output quality metrics
- **Execution Time** — Total pipeline latency
- **API Calls Count** — Number of LLM invocations
- **Iteration Count** — How many refinement loops were needed
- **System Efficiency Score** — Composite metric combining all factors

Example output:
```
=== Final Output ===
[Complete, approved response]

=== System Evaluation ===
{
  'execution_time': 12.34,
  'worker_calls': 2,
  'reviewer_calls': 3,
  'revision_count': 1,
  'system_score': 8.2/10
}
```

---

## 🎓 Key Concepts & Patterns

### 1. Action Trace Pattern
Every agent decision is logged with context:
```python
action_trace = [
    {
        "action": "search_tool",
        "input": "query text",
        "result": "search results"
    },
    ...
]
```

### 2. State Management with LangGraph
Agents pass and modify shared state dictionary:
```python
state = {
    "user_query": "...",
    "plan": "...",
    "draft_response": "...",
    "behavior_feedback": "...",
    "reasoning_feedback": "...",
    "review_reason": "...",
    "revision_count": 0
}
```

### 3. Conditional Routing
The router function uses state to make decisions:
```python
def router(state):
    if all_approvals(state) or max_iterations_reached(state):
        return "__end__"
    return "worker"  # Send back for refinement
```

### 4. Feedback Loops
Failed evaluations pass structured feedback:
```python
feedback = (
    state["behavior_feedback"] + "\n" +
    state["reasoning_feedback"] + "\n" +
    state["review_reason"]
)
# Worker uses this for next iteration
```

---

## 💡 Production Considerations

### Scaling
- Use async/await for parallel evaluations
- Implement caching for repeated queries
- Use vector DB for semantic search

### Cost Optimization
- Track token usage per agent
- Implement early exit for clear approvals
- Use cheaper models for secondary evaluations

### Robustness
- Implement retry logic for API failures
- Add timeout handling
- Log all failures for analysis

### Monitoring
- Track approval rates per evaluation layer
- Monitor average iterations needed
- Measure cost per successful completion

---

## 🧪 Testing & Benchmarking

Run on diverse query types to benchmark the system:

```bash
# Test queries
python system_eval.py
# → "Explain quantum computing"

python system_eval.py
# → "Write a Python decorator for caching"

python system_eval.py
# → "Compare REST vs GraphQL APIs"
```

Monitor the logs to understand:
- Which evaluation layer is most critical for your use case
- Typical iteration counts needed
- Cost-quality trade-offs
- Bottlenecks in the pipeline

---

## 📚 Learning Path

1. **Start with `app.py`** — Understand the basic 3-layer pipeline
2. **Study `system_eval.py`** — See how to add behavior evaluation
3. **Explore `reasoning_eval.py`** — Learn reasoning quality assessment
4. **Read `logs/`** — Analyze real execution traces
5. **Modify & Extend** — Build your own evaluation layers

---

## 🤝 Contributing

This is a educational framework. Fork it, modify it, and build your own evaluation systems!

---

## 📖 References & Further Reading

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph/)
- [Groq API Docs](https://console.groq.com)
- [LLM-as-a-Judge Pattern](https://arxiv.org/abs/2310.05470)
- [Agentic AI Evaluation](https://arxiv.org/abs/2308.00228)

---

## 📝 License

MIT License — Feel free to use for research, education, and production systems.

---

**Built with ❤️ for AI practitioners who believe evaluation is the missing link in AI quality.**
