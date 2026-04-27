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

if "run_clicked" not in st.session_state:
    st.session_state["run_clicked"] = False

if "overview" not in st.session_state:
    st.session_state["overview"] = None

if "report" not in st.session_state:
    st.session_state["report"] = None

if "model_explain" not in st.session_state:
    st.session_state["model_explain"] = None


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

    tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Model", "🤖 AI Chat"])

    # ===============================
    # TAB 1: DATA
    # ===============================
    with tab1:

        st.markdown("### 📊 Dataset Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("🔍 Preview Data"):
            st.dataframe(df.head(100), use_container_width=True)

        st.divider()

        st.markdown("### ⚙️ Train Model")

        target_column = st.selectbox("🎯 Select Target Column", df.columns)

        if st.button("🚀 Run Pipeline"):
            st.session_state["run_clicked"] = True
            st.session_state["results"] = None
            st.session_state["overview"] = None
            st.session_state["report"] = None
            st.session_state["model_explain"] = None

        if st.session_state.get("run_clicked", False):

            from core.pipeline import run_pipeline
            from utils.report_generator import generate_advanced_report

            # 🔥 Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_box = st.empty()

            logs = []

            def progress_callback(percent, message):
                progress_bar.progress(percent)

                # Avoid duplicate logs
                if not logs or logs[-1] != message:
                    logs.append(message)

                # Show current step ONLY here
                status_text.markdown(f"### 🔄 {message}")

                # Show ONLY previous steps (muted)
                previous_logs = logs[:-1]

                styled_logs = [
                    f"<span style='color:gray'>{log}</span>"
                    for log in previous_logs[-6:]
                ]

                log_box.markdown("<br>".join(styled_logs), unsafe_allow_html=True)

            # 🔥 Run pipeline with progress tracking
            results = run_pipeline(
                df,
                target_column,
                progress_callback=progress_callback
            )

            st.session_state["results"] = results
            st.session_state["chat_history"] = []

            # 🔥 Generate Report
            if results.get("success", False):
                st.session_state["report"] = generate_advanced_report(results, df)

            # 🔥 Clear progress UI
            progress_bar.empty()
            status_text.empty()

            st.session_state["run_clicked"] = False

        results = st.session_state.get("results", None)

        if results:
            if not results.get("success", False):
                st.error(results.get("error", "Unknown error"))
            else:
                st.success("✅ Pipeline Completed")

    # ===============================
    # TAB 2: MODEL
    # ===============================
    with tab2:

        results = st.session_state.get("results", None)

        if results and results.get("success", False):

            # 🔥 SUBTABS
            subtab1, subtab2 = st.tabs(["📊 Results", "📄 Report Center"])

            # ============================
            # 📊 SUBTAB 1: RESULTS
            # ============================
            with subtab1:

                st.markdown("## 📊 Model Results")

                col1, col2 = st.columns(2)
                col1.info(f"Task: {results['task_type']}")
                col2.success(f"Model: {results['model']}")

                st.markdown("### 📈 Performance")
                metric_cols = st.columns(len(results["metrics"]))

                for i, (k, v) in enumerate(results["metrics"].items()):
                    metric_cols[i].metric(k, round(v, 4))

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

                # 🔍 ADVANCED INSIGHTS
                st.divider()
                st.markdown("## 🔍 Advanced Insights")

                with st.expander("📊 Data Summary"):
                    st.write(df.describe())

                with st.expander("🧾 Pipeline Logs"):
                    for log in results.get("logs", []):
                        st.text(log)

            # ============================
            # 📄 SUBTAB 2: REPORT CENTER
            # ============================
            with subtab2:

                st.markdown("## 📄 Advanced Report Center")

                report = st.session_state.get("report", None)

                if report:

                    with st.expander("📖 Preview Report"):
                        st.text_area("Report", report, height=400)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            label="📥 Download TXT Report",
                            data=report,
                            file_name="ml_advanced_report.txt",
                            mime="text/plain"
                        )

                    with col2:
                        st.download_button(
                            label="📄 Download Markdown",
                            data=report,
                            file_name="ml_report.md",
                            mime="text/markdown"
                        )

                else:
                    st.info("Run the pipeline to generate report")

        else:
            st.info("Run the pipeline in Data tab to see results")

    # ===============================
    # TAB 3: AI CHAT
    # ===============================
    with tab3:

        results = st.session_state.get("results", None)

        if results and results.get("success", False):

            from llm_layer.llm_agent import LLMAgent
            from llm_layer.prompt_templates import (
                get_explanation_prompt,
                get_chat_prompt
            )

            agent = LLMAgent()

            st.markdown("## 🤖 AI Assistant")

            if st.session_state["overview"] is None:
                with st.spinner("Generating AI insights..."):
                    prompt = get_explanation_prompt(results)
                    st.session_state["overview"] = agent.call_llm(prompt)

            with st.expander("📘 AI Overview", expanded=False):
                st.write(st.session_state["overview"])

            st.divider()

            st.markdown("### 💬 Chat")

            MAX_VISIBLE = 10
            visible_chats = st.session_state["chat_history"][-MAX_VISIBLE:]

            for chat in visible_chats:
                st.write(f"🧑 {chat['user']}")
                st.write(f"🤖 {chat['bot']}")

            if st.button("🧹 Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()

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
            st.info("Run the pipeline first from Data tab to enable AI assistant")