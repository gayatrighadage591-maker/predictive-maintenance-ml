import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_models():
    logistic = joblib.load("logistic_regression_model.pkl")
    forest = joblib.load("random_forest_model.pkl")
    features = joblib.load("feature_columns.pkl")
    return logistic, forest, features

@st.cache_data
def load_dataset():
    return pd.read_csv("prepared_predictive_maintenance.csv")

try:
    logistic_model, forest_model, feature_columns = load_models()
    df = load_dataset()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Predictive Maintenance")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Dataset Overview",
        "Data Visualization",
        "Failure Prediction",
        "Model Comparison",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Mini Project")

# ---------------- HOME ----------------

if page == "Home":

    st.title("⚙️ Predictive Maintenance System")

    st.subheader(
        "Machine Learning-Based Predictive Maintenance "
        "and Machine Failure Risk Prediction System"
    )

    st.write(
        "This project uses machine learning to analyze machine "
        "operating conditions and predict potential machine failure."
    )

    st.info(
        "Enter machine operating parameters to obtain a prediction "
        "from the trained machine learning model."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", f"{len(df):,}")
    col2.metric("Input Features", len(feature_columns))
    col3.metric("ML Algorithms", 2)

    st.markdown("### Project Objectives")

    st.markdown("""
    - Analyze machine operating conditions.
    - Explore machine failure patterns.
    - Train Logistic Regression and Random Forest models.
    - Compare model performance using classification metrics.
    - Provide an interactive machine failure prediction interface.
    """)

# ---------------- DATASET OVERVIEW ----------------

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.write(
        "Dataset: AI4I 2020 Predictive Maintenance Dataset"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Column Information")
    st.write(df.dtypes.astype(str).to_frame("Data Type"))

# ---------------- DATA VISUALIZATION ----------------

elif page == "Data Visualization":

    st.title("📈 Data Visualization")

    st.subheader("Machine Failure Distribution")

    failure_counts = df["Machine failure"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["No Failure", "Failure"],
        failure_counts.values
    )
    ax.set_ylabel("Number of Records")
    ax.set_title("Machine Failure Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Operating Parameter Distribution")

    parameter = st.selectbox(
        "Select a parameter",
        feature_columns
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[parameter], bins=30)
    ax.set_xlabel(parameter)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {parameter}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

# ---------------- FAILURE PREDICTION ----------------

elif page == "Failure Prediction":

    st.title("🤖 Machine Failure Prediction")

    st.write(
        "Enter machine operating parameters to predict "
        "whether machine failure is likely."
    )

    st.warning(
        "Predictions are based on the trained model and are "
        "not a guarantee of actual machine behavior."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1
            )

            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=100,
                max_value=10000,
                value=1500
            )

        with col2:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.1
            )

            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=500,
                value=100
            )

            model_choice = st.selectbox(
                "Choose Prediction Model",
                ["Random Forest", "Logistic Regression"]
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                air_temp,
                process_temp,
                rotational_speed,
                torque,
                tool_wear
            ]],
            columns=[
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
        )

        if list(input_data.columns) != list(feature_columns):
            st.error(
                "The model feature order does not match the input form."
            )
            st.stop()

        model = (
            forest_model
            if model_choice == "Random Forest"
            else logistic_model
        )

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]
        failure_probability = float(
            probabilities[list(model.classes_).index(1)]
        )

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Machine Failure Predicted")
        else:
            st.success("✅ No Machine Failure Predicted")

        st.metric(
            "Estimated Failure Probability",
            f"{failure_probability:.1%}"
        )

        st.caption(
            "This probability is a model estimate, not a calibrated "
            "guarantee of the actual failure risk."
        )

# ---------------- MODEL COMPARISON ----------------

elif page == "Model Comparison":

    st.title("📋 Model Performance Comparison")

    X_test_features = df[feature_columns]
    y_true = df["Machine failure"]

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": forest_model
    }

    result_rows = []

    for name, model in models.items():

        predictions = model.predict(X_test_features)

        result_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, predictions),
            "Precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "F1-Score": f1_score(
                y_true, predictions, zero_division=0
            )
        })

    comparison_df = pd.DataFrame(result_rows)

    st.dataframe(
        comparison_df.set_index("Model").style.format("{:.3f}"),
        use_container_width=True
    )

    st.warning(
        "These metrics are calculated on the prepared dataset. "
        "For unbiased evaluation, use the held-out test set from "
        "your training notebook instead."
    )

    selected_metric = st.selectbox(
        "Select a metric to compare",
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        comparison_df["Model"],
        comparison_df[selected_metric]
    )
    ax.set_ylabel(selected_metric)
    ax.set_ylim(0, 1)
    ax.set_title(f"Model Comparison: {selected_metric}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Confusion Matrices")

    for name, model in models.items():

        predictions = model.predict(X_test_features)

        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(name)

        st.pyplot(fig)
        plt.close(fig)

# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.subheader("Problem Statement")

    st.write(
        "Unexpected machine failures can interrupt industrial "
        "operations. This project explores machine operating data "
        "to predict recorded machine failures using classification."
    )

    st.subheader("Methodology")

    st.markdown("""
    1. Dataset collection.
    2. Data inspection and preprocessing.
    3. Exploratory data analysis.
    4. Training Logistic Regression and Random Forest.
    5. Model evaluation and comparison.
    6. Deployment through Streamlit.
    """)

    st.subheader("Applications")

    st.markdown("""
    - Industrial equipment monitoring.
    - Maintenance planning support.
    - Analysis of machine operating conditions.
    - Educational demonstration of predictive maintenance.
    """)

    st.subheader("Limitations")

    st.markdown("""
    - Predictions depend on the quality and limitations of the dataset.
    - The application uses manually entered parameters.
    - The model does not receive live sensor readings.
    - Predictions require validation before real industrial use.
    """)

    st.subheader("Future Scope")

    st.markdown("""
    - Integrate real-time sensor data.
    - Evaluate additional machine learning algorithms.
    - Explore failure-type prediction.
    - Improve model calibration and monitoring.
    """)

    st.subheader("Dataset Source")

    st.markdown(
        "[AI4I 2020 Predictive Maintenance Dataset — UCI Repository]"
        "(https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)"
    )

    st.caption(
        "Team Details: Add your group members' names, roll numbers, "
        "class, and project guide before submission."
    )

# ---------------- FOOTER ----------------

st.sidebar.markdown("---")
st.sidebar.caption("Predictive Maintenance ML Project")
