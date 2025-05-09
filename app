import streamlit as st
import pandas as pd
import datetime

from utils import (
    validate_csv,
    detect_prompt_injection,
    generate_column_summary_table,
    suggest_fixes,
    TOOLS
)
from agent import run_agent_pipeline

st.set_page_config(page_title="Interactive Data Cleaner", layout="wide")
st.title("🧠 Interactive Data Cleaner Agent")

for key in ["df", "suggested_tools", "cleaned_df", "feedback_history", "log"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "feedback_history" else []

if st.button("🧹 Reset"):
    for key in ["df", "suggested_tools", "cleaned_df", "feedback_history", "file_uploader", "log"]:
        st.session_state[key] = None
    st.session_state.clear()
    st.rerun()

file = st.file_uploader("📂 Upload your CSV", type=["csv"], key="file_uploader")
if file:
    try:
        if file.size > 200 * 1024 * 1024:
            st.error("❌ File is too large (limit is 200MB).")
            st.stop()

        df = pd.read_csv(file)
        issues = validate_csv(df) + detect_prompt_injection(df)

        if issues:
            for issue in issues:
                st.error(f"❌ {issue}")
            st.stop()

        st.session_state.df = df

        st.subheader("📄 Original Data")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("### 📊 Column Summary for Evaluation")
        summary_df = generate_column_summary_table(df)
        st.dataframe(summary_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

if st.session_state["df"] is not None:
    if st.button("🧠 Analyze & Suggest Cleaning Steps"):
        st.session_state["suggested_tools"] = suggest_fixes(st.session_state["df"])

if st.session_state["suggested_tools"]:
    st.subheader("🔧 Suggested Cleaning Steps")
    all_tool_options = list(TOOLS.keys())
    tool_set = set(all_tool_options) | set(st.session_state["suggested_tools"])
    combined_tool_options = sorted(tool_set)

    selected_tools = st.multiselect(
        "Review, remove, or add tools below before running:",
        options=combined_tool_options,
        default=[t for t in st.session_state["suggested_tools"] if t in combined_tool_options],
        help="Only selected tools will be run. You can add or remove freely."
    )

    if st.button("🚀 Run Cleaner"):
        with st.spinner("Agent cleaning in progress..."):
            cleaned, log = run_agent_pipeline(st.session_state["df"], tools=selected_tools)
            st.session_state["cleaned_df"] = cleaned
            st.session_state["log"] = log
        st.success("✅ Cleaning complete.")

if st.session_state["cleaned_df"] is not None:
    st.subheader("📦 Final Cleaned Data")
    st.dataframe(st.session_state["cleaned_df"].head(100), use_container_width=True)

    st.markdown("### 📊 Updated Column Summary")
    summary_df = generate_column_summary_table(st.session_state["cleaned_df"])
    st.dataframe(summary_df, use_container_width=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cleaned_{timestamp}.csv"
    st.download_button(
        "⬇ Download Cleaned CSV",
        st.session_state["cleaned_df"].to_csv(index=False),
        file_name=filename,
        mime="text/csv"
    )

    st.markdown("### 📝 Cleaning Log")
    for log_entry in st.session_state.get("log", []):
        st.write(log_entry)

    st.markdown("### 🗣️ Provide Feedback to Improve Cleaning")
    feedback = st.text_area("Still see issues? Describe them:")

    if st.button("🔁 Re-run With Feedback"):
        if feedback.strip():
            st.session_state["feedback_history"].append(feedback.strip())
        combined_feedback = "\n".join(st.session_state["feedback_history"])
        with st.spinner("Agent re-cleaning in progress..."):
            re_cleaned, re_log = run_agent_pipeline(
                st.session_state["cleaned_df"],
                tools=selected_tools,
                feedback=combined_feedback)
            st.session_state["cleaned_df"] = re_cleaned
            st.session_state["log"] += re_log
        st.success("✅ Agent re-cleaning complete.")
        st.rerun()
