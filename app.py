import streamlit as st
from agent.graph import build_graph

st.set_page_config(page_title="Vacation Planner Agent", layout="wide")

st.title("Vacation Planner Agent (LangGraph + Tools + Memory)")
st.caption("Agentic workflow: parse → propose → choose → plan → budget → adjust → finalize")

default_prompt = (
    "Plan a 5-day trip in March for 2 people, budget $1800, "
    "I like food and museums, medium pace. Prefer Europe."
)

user_request = st.text_area("Describe your trip request", value=default_prompt, height=120)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Run Agent", type="primary")
with col2:
    show_trace = st.checkbox("Show agent trace", value=True)

if run_btn:
    graph = build_graph()
    out = graph.invoke({"user_request": user_request})

    st.subheader("Final Output")
    st.markdown(out["final_plan"])

    if show_trace:
        st.subheader("Trace (what the agent did)")
        for i, h in enumerate(out.get("history", []), start=1):
            with st.expander(f"Step {i}: {h.get('node', 'unknown')}"):
                st.json(h, expanded=False)

        st.subheader("Structured State")
        st.json({
            "constraints": out.get("constraints"),
            "selected_destination": out.get("selected_destination"),
            "budget": out.get("budget"),
        })
