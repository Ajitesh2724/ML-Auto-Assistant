import streamlit as st
import pandas as pd

# MUST be first Streamlit command
st.set_page_config(
    page_title="ML Auto Assistant",
    layout="wide"
)

st.title("🤖 ML Auto Assistant")
st.write("Upload your dataset and let AI build your ML pipeline")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    # -------------------------------
    # TARGET COLUMN
    # -------------------------------
    target_column = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )

    # -------------------------------
    # RUN BUTTON
    # -------------------------------
    if st.button("🚀 Run AutoML Pipeline"):

        try:
            from core.pipeline import run_pipeline

            with st.spinner("Running full ML pipeline..."):

                results = run_pipeline(
                    df,
                    target_column,
                    task_type="auto"
                )

            # -------------------------------
            # HANDLE RESULT
            # -------------------------------
            if not results["success"]:
                st.error(f"❌ Error: {results['error']}")
            else:
                st.success("✅ Pipeline Completed Successfully!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📌 Task Type")
                    st.write(results["task_type"])

                with col2:
                    st.subheader("🏆 Selected Model")
                    st.write(results["model"])

                st.subheader("📈 Metrics")
                st.json(results["metrics"])

                st.subheader("🧠 Selected Features")
                st.write(results["features"])

        except Exception as e:
            st.error(f"🔥 App Crash: {e}")