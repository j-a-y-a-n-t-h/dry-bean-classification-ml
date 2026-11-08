
"""
Dry Bean Classification Dashboard
Loads saved models only; uploaded CSV is used for evaluation, never for training.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification Dashboard",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1 {
    font-size: 2rem !important;
    letter-spacing: -0.5px;
}

[data-testid="stMetric"] {
    background: #202938 !important;
    border: 1px solid #3a4658 !important;
    padding: 13px 15px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
    font-size: 0.82rem;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #f8fafc !important;
    font-size: 1.65rem;
}

div[data-testid="stSelectbox"] {
    max-width: 460px;
}

.small-note {
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MODEL_DIR = "model"
TARGET = "Class"


# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    metadata_path = os.path.join(
        MODEL_DIR,
        "metadata.json"
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Model metadata not found. Run train_models.py first."
        )

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    loaded_models = {}

    for display_name, filename in MODEL_FILES.items():

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing model file: {model_path}"
            )

        loaded_models[display_name] = joblib.load(
            model_path
        )

    return metadata, loaded_models


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("🫘 Dry Bean Classification")

st.caption(
    "Model evaluation dashboard · 7 bean varieties · 16 numerical features"
)

st.markdown(
    "Upload a labelled test CSV to evaluate the saved classifiers. "
    "**Models are loaded from disk and are never retrained by this app.**"
)


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

with st.container(border=True):

    uploaded = st.file_uploader(
        "Upload labelled test data (CSV)",
        type=["csv"],
    )

    st.caption(
        "Required: all 16 input features and the true `Class` column."
    )


# --------------------------------------------------
# LOAD MODELS AND METADATA
# --------------------------------------------------

try:

    metadata, models = load_artifacts()

except Exception as exc:

    st.error(
        f"Could not load trained models: {exc}"
    )

    st.stop()


if uploaded is None:

    st.info(
        "Upload the generated `test_data.csv` to view the evaluation dashboard."
    )

    st.markdown(
        "**Available classifiers:** "
        + " · ".join(MODEL_FILES.keys())
    )

    st.stop()


# --------------------------------------------------
# EVALUATE UPLOADED TEST DATA
# --------------------------------------------------

try:

    test_df = pd.read_csv(uploaded)

    feature_columns = metadata["feature_columns"]

    known_classes = metadata["class_labels"]

    required_columns = feature_columns + [TARGET]

    # Check required columns

    missing_columns = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing required columns: {missing_columns}"
        )

        st.stop()

    # Check empty dataset

    if test_df.empty:

        st.error(
            "The uploaded CSV is empty."
        )

        st.stop()

    # Check missing values

    if test_df[required_columns].isna().any().any():

        st.error(
            "The uploaded test data contains missing values."
        )

        st.stop()

    # Separate features and target

    X_test = test_df[feature_columns]

    y_test = test_df[TARGET].astype(str)

    # Check numeric features

    if not all(
        pd.api.types.is_numeric_dtype(X_test[column])
        for column in X_test.columns
    ):

        st.error(
            "All input features must be numeric."
        )

        st.stop()

    # Check target labels

    unexpected_classes = sorted(
        set(y_test.unique()) - set(known_classes)
    )

    if unexpected_classes:

        st.error(
            f"Unknown target labels in uploaded CSV: {unexpected_classes}"
        )

        st.stop()

    # --------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------

    left, right = st.columns(
        [1.15, 2.2],
        vertical_alignment="bottom"
    )

    with left:

        model_name = st.selectbox(
            "Choose classifier",
            list(MODEL_FILES.keys())
        )

    with right:

        st.markdown(
            f"<div class='small-note'>"
            f"Evaluating <b>{len(test_df):,}</b> test records · "
            f"<b>{len(known_classes)}</b> target classes"
            f"</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    model = models[model_name]

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    # --------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------

    present_classes = sorted(
        y_test.unique().tolist()
    )

    if len(present_classes) == len(known_classes):

        auc = roc_auc_score(
            y_test,
            probabilities,
            labels=known_classes,
            multi_class="ovr",
            average="macro",
        )

    else:

        auc = None

    metric_values = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "AUC · Macro OVR": auc,

        "Precision · Macro": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "Recall · Macro": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "F1 · Macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y_test,
            predictions
        ),
    }

    # --------------------------------------------------
    # DISPLAY METRICS
    # --------------------------------------------------

    st.subheader(
        f"{model_name} · Performance"
    )

    metric_columns = st.columns(6)

    for column, (metric, value) in zip(
        metric_columns,
        metric_values.items()
    ):

        column.metric(
            metric,
            "N/A" if value is None else f"{value:.4f}"
        )

    if auc is None:

        st.warning(
            "AUC is unavailable because the uploaded test file "
            "does not contain all seven classes."
        )

    st.divider()

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------

    matrix_tab, report_tab, data_tab = st.tabs(
        [
            "Confusion matrix",
            "Classification report",
            "Test data preview",
        ]
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    with matrix_tab:

        st.caption(
            "Rows = actual class · Columns = predicted class · "
            "Diagonal cells are correct predictions."
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=known_classes,
        )

        fig, ax = plt.subplots(
            figsize=(7.2, 5.2),
            dpi=110,
        )

        ax.imshow(
            cm,
            cmap="Blues",
            interpolation="nearest",
        )

        ax.set(
            xticks=np.arange(len(known_classes)),
            yticks=np.arange(len(known_classes)),
            xticklabels=known_classes,
            yticklabels=known_classes,
            xlabel="Predicted class",
            ylabel="Actual class",
            title="",
        )

        plt.setp(
            ax.get_xticklabels(),
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=8,
        )

        plt.setp(
            ax.get_yticklabels(),
            fontsize=8,
        )

        threshold = cm.max() / 2.0

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[i, j] > threshold
                        else "#172554"
                    ),
                    fontsize=8,
                    fontweight="semibold",
                )

        ax.set_xticks(
            np.arange(-.5, len(known_classes), 1),
            minor=True,
        )

        ax.set_yticks(
            np.arange(-.5, len(known_classes), 1),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linestyle="-",
            linewidth=1.2,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        fig.tight_layout(pad=1.0)

        st.pyplot(
            fig,
            use_container_width=False,
        )

        plt.close(fig)

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    with report_tab:

        report = classification_report(
            y_test,
            predictions,
            labels=known_classes,
            zero_division=0,
            output_dict=True,
        )

        report_df = pd.DataFrame(
            report
        ).transpose().round(4)

        st.dataframe(
            report_df,
            use_container_width=True,
            height=360,
        )

    # --------------------------------------------------
    # TEST DATA PREVIEW
    # --------------------------------------------------

    with data_tab:

        st.dataframe(
            test_df.head(30),
            use_container_width=True,
            height=420,
        )

    # --------------------------------------------------
    # COMPARE ALL FIVE MODELS
    # --------------------------------------------------

    if os.path.exists("metrics.csv"):

        with st.expander(
            "Compare all five trained models"
        ):

            comparison = pd.read_csv(
                "metrics.csv"
            )

            display_columns = [
                column
                for column in [
                    "Model",
                    "Accuracy",
                    "AUC",
                    "Precision_macro",
                    "Recall_macro",
                    "F1_macro",
                    "MCC",
                ]
                if column in comparison.columns
            ]

            numeric_columns = [
                column
                for column in display_columns
                if column != "Model"
            ]

            # Round values without using pandas Styler.
            # This avoids the Jinja2 dependency.

            comparison[numeric_columns] = (
                comparison[numeric_columns].round(4)
            )

            st.dataframe(
                comparison[display_columns],
                use_container_width=True,
                hide_index=True,
            )

except Exception as exc:

    st.error(
        f"Could not evaluate uploaded CSV: {exc}"
    )
