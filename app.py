import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ML Auto Assistant", layout="wide")

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<h1 style='text-align: center;'>🤖 ML Auto Assistant</h1>
<p style='text-align: center; color: gray;'>AutoML + AI Insights Dashboard</p>
""", unsafe_allow_html=True)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # DATA OVERVIEW (TAB 1)
    # -------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Model", "🤖 AI Chat"])

    with tab1:

        st.markdown("### 📊 Dataset Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("🔍 Preview Data"):
            st.dataframe(df.head(100), use_container_width=True)

    # -------------------------------
    # TARGET SELECTION
    # -------------------------------
    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    # -------------------------------
    # RUN PIPELINE
    # -------------------------------
    if st.button("🚀 Run Pipeline"):

        from core.pipeline import run_pipeline

        with st.spinner("Training model..."):
            results = run_pipeline(df, target_column)

        st.session_state["results"] = results
        st.session_state["chat_history"] = []

    results = st.session_state.get("results", None)
    if results:

            if not results["success"]:
                st.error(results["error"])

            else:
                st.success("✅ Pipeline Completed")

    # ===============================
    # MODEL TAB
    # ===============================
    with tab2:

        if results:

            if not results["success"]:
                st.error(results["error"])

            else:
                

                st.markdown("## 📊 Model Results")

                col1, col2 = st.columns(2)
                col1.info(f"Task: {results['task_type']}")
                col2.success(f"Model: {results['model']}")

                # Metrics
                st.markdown("### 📈 Performance")
                metric_cols = st.columns(len(results["metrics"]))
                for i, (k, v) in enumerate(results["metrics"].items()):
                    metric_cols[i].metric(k, round(v, 4))

                # Feature Importance
                st.markdown("### 🧠 Feature Importance")

                if results.get("feature_importance"):

                    importance_df = pd.DataFrame(results["feature_importance"])
                    importance_df = importance_df.sort_values(by="importance", ascending=False)

                    fig = px.bar(
                        importance_df.head(15),
                        x="importance",
                        y="feature",
                        orientation="h"
                    )

                    fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No feature importance available")

                with st.expander("📌 Selected Features"):
                    st.write(results["features"])

        else:
            st.info("Run the pipeline to see results")

    # ===============================
    # AI CHAT TAB
    # ===============================
    with tab3:

        if results:

            from llm_layer.llm_agent import LLMAgent
            from llm_layer.prompt_templates import (
                get_explanation_prompt,
                get_chat_prompt
            )

            agent = LLMAgent()

            st.markdown("## 🤖 AI Assistant")

            # -------------------------------
            # COLLAPSIBLE OVERVIEW
            # -------------------------------
            with st.expander("📘 AI Overview", expanded=False):

                with st.spinner("Generating insights..."):
                    prompt = get_explanation_prompt(results)
                    explanation = agent.call_llm(prompt)

                st.write(explanation)

            st.divider()

            # -------------------------------
            # CHAT WINDOW (LIMITED)
            # -------------------------------
            st.markdown("### 💬 Chat")

            MAX_VISIBLE = 10
            visible_chats = st.session_state["chat_history"][-MAX_VISIBLE:]

            for chat in visible_chats:
                st.write(f"🧑 {chat['user']}")
                st.write(f"🤖 {chat['bot']}")

            # -------------------------------
            # CLEAR CHAT
            # -------------------------------
            if st.button("🧹 Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()

            # -------------------------------
            # CHAT INPUT
            # -------------------------------
            with st.form("chat_form", clear_on_submit=True):

                user_q = st.text_input("Ask about your model...")

                submitted = st.form_submit_button("Send")

                if submitted and user_q:

                    with st.spinner("Thinking..."):
                        chat_prompt = get_chat_prompt(results, user_q)
                        answer = agent.call_llm(chat_prompt)

                    st.session_state["chat_history"].append({
                        "user": user_q,
                        "bot": answer
                    })

                    st.rerun()

        else:
            st.info("Run the pipeline first to enable AI assistant")