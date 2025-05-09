import pandas as pd
from typing import List, Optional, TypedDict

from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

from utils import TOOLS, verify_tool_effect

llm = ChatOpenAI(model="gpt-4", temperature=0)

class CleaningState(TypedDict):
    df: pd.DataFrame
    actions_taken: List[str]
    feedback: str
    tool_decision: Optional[str]
    available_tools: List[str]
    step_count: int
    log: List[str]

def apply_tool(state: CleaningState) -> CleaningState:
    tool = state["tool_decision"]
    df = state["df"]

    if tool not in TOOLS:
        state["log"].append(f"❌ Tool '{tool}' not found.")
        return state

    try:
        before_df = df.copy()
        new_df = TOOLS[tool](before_df.copy())
        changed = verify_tool_effect(before_df, new_df, tool)

        if changed:
            state["df"] = new_df
            state["actions_taken"].append(tool)
            state["log"].append(f"✅ Applied: {tool}")
        else:
            state["actions_taken"].append(f"{tool} - no effect")
            state["log"].append(f"⚠️ Tool '{tool}' had no effect.")

    except Exception as e:
        state["actions_taken"].append(f"Failed: {tool} -> {str(e)}")
        state["log"].append(f"❌ Tool '{tool}' failed: {str(e)}")

    return state

def choose_tool(state: CleaningState) -> CleaningState:
    sample = state["df"].sample(min(20, len(state["df"])))
    prompt = f"""
You are a data cleaning assistant.

## Dataset Sample
{sample.to_string(index=False)}

## Cleaning History
{state['actions_taken']}

## User Feedback
{state['feedback']}

## Available Tools
{state['available_tools']}

Choose the best next tool to apply from the list.
Respond with only the tool name (e.g., remove_empty_rows), or 'end' to stop.
"""
    try:
        response = llm.invoke(prompt).content.strip().lower().strip("'\"")
        if response not in state["available_tools"] and response != "end":
            state["log"].append(f"⚠️ Ignoring invalid tool: {response}")
            state["tool_decision"] = "end"
        else:
            state["tool_decision"] = response
    except Exception as e:
        state["log"].append(f"❌ Error in tool decision: {e}")
        state["tool_decision"] = "end"

    state["step_count"] += 1
    if state["step_count"] > 10:
        state["log"].append("⚠️ Max steps reached. Stopping.")
        state["tool_decision"] = "end"

    return state

workflow = StateGraph(CleaningState)
workflow.add_node("choose_tool", choose_tool)
workflow.add_node("apply_tool", apply_tool)
workflow.add_node("end", lambda s: s)

workflow.set_entry_point("choose_tool")
workflow.add_edge("choose_tool", "apply_tool")
workflow.add_conditional_edges(
    "apply_tool",
    lambda s: END if s["tool_decision"] == "end" else "choose_tool"
)

graph = workflow.compile()

def run_agent_pipeline(df: pd.DataFrame, tools: List[str] = None, feedback: str = ""):
    initial_state: CleaningState = {
        "df": df,
        "actions_taken": [],
        "feedback": feedback,
        "available_tools": tools or list(TOOLS.keys()),
        "tool_decision": None,
        "step_count": 0,
        "log": []
    }
    final_state = graph.invoke(initial_state)
    if "__end__" in final_state:
        final_state = final_state["__end__"]
    return final_state["df"], final_state["log"]
