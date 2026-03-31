import streamlit as st
import pandas as pd
import plotly.express as px


# must be first streamlit command
st.set_page_config(
    page_title="ML Auto Assistant",
    layout="wide"
)

st.title("🤖 ML Auto Assistant")

st.write(
    "Upload dataset and automatically build ML model"
)


# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)


# -------------------------------
# MAIN UI
# -------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.write("Shape:", df.shape)


    # -------------------------------
    # TARGET COLUMN
    # -------------------------------
    target_column = st.selectbox(

        "Select Target Column",

        df.columns
    )


    # -------------------------------
    # RUN BUTTON
    # -------------------------------
    if st.button("Run Pipeline"):

        from core.pipeline import run_pipeline

        with st.spinner("Training model..."):

            results = run_pipeline(

                df,

                target_column,

                task_type="auto"
            )


        # -------------------------------
        # HANDLE ERRORS
        # -------------------------------
        if not results["success"]:

            st.error(results["error"])


        else:

            st.success("Pipeline completed")


            # -------------------------------
            # SUMMARY
            # -------------------------------
            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Task Type")

                st.write(

                    results["task_type"]
                )


            with col2:

                st.subheader("Selected Model")

                st.write(

                    results["model"]
                )


            # -------------------------------
            # METRICS
            # -------------------------------
            st.subheader("Metrics")

            st.json(

                results["metrics"]
            )


            # -------------------------------
            # FEATURE IMPORTANCE
            # -------------------------------
            st.subheader("Feature Importance")


            if results["feature_importance"] is not None:

                importance_df = pd.DataFrame(

                    results["feature_importance"]
                )


                importance_df = importance_df.sort_values(

                    by="importance",

                    ascending=False
                )


                fig = px.bar(

                    importance_df.head(20),

                    x="importance",

                    y="feature",

                    orientation="h",

                    title="Top Features"
                )


                fig.update_layout(

                    height=500,

                    yaxis=dict(

                        autorange="reversed"
                    )
                )


                st.plotly_chart(

                    fig,

                    use_container_width=True
                )


            else:

                st.info(

                    "Model does not provide feature importance"
                )


            # -------------------------------
            # FEATURE LIST
            # -------------------------------
            st.subheader("Selected Features")


            st.write(

                results["features"]
            )