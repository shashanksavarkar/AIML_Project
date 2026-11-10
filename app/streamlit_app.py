"""Streamlit demo: pick an example wafer (or upload one) and get a predicted defect category.

Run from the project folder:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]      # the project folder (app/ sits one level below it)
sys.path.append(str(ROOT))                      # lets Python find src/features.py
from src.features import extract_features       # the SAME feature code that was used for training

MODEL_FILE = ROOT / "models" / "wafer_defect_rf.joblib"
SAMPLE_FILE = ROOT / "data" / "processed" / "demo_samples.joblib"
REPORTS = ROOT / "reports"

# Colours: 0 = off-wafer (grey), 1 = good die (blue), 2 = defective die (red)
WAFER_CMAP = ListedColormap(["#f0f0f0", "#9ecae1", "#d62728"])


@st.cache_resource                              # load the model once, not on every click
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data                                  # load the example wafers once
def load_samples():
    return joblib.load(SAMPLE_FILE)


def draw_wafer(wafer):
    """Draw a wafer map as a picture."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.asarray(wafer), cmap=WAFER_CMAP, vmin=0, vmax=2)
    ax.axis("off")
    return fig


def predict(bundle, wafer):
    """Return a table of class probabilities, highest first, for one wafer map."""
    feats = pd.DataFrame([extract_features(wafer)])[bundle["feature_names"]]   # same 15 features, same order
    probs = bundle["model"].predict_proba(feats)[0]                            # one probability per class
    names = [bundle["class_names"][c] for c in bundle["model"].classes_]       # class index -> name
    table = pd.Series(probs, index=names).sort_values(ascending=False)
    return table


def read_uploaded(file):
    """Read an uploaded .npy or .csv/.txt file into a 2D grid of 0, 1, 2 values."""
    if file.name.endswith(".npy"):
        grid = np.load(file)
    else:
        grid = np.loadtxt(file, delimiter=",")
    if grid.ndim != 2:
        raise ValueError("The file must contain a 2D grid of numbers (rows and columns).")
    if not set(np.unique(grid)).issubset({0, 1, 2}):
        raise ValueError("Pixel values must be 0 (off-wafer), 1 (good die) or 2 (defective die).")
    return grid


# ---------- Page ----------
st.set_page_config(page_title="Wafer Defect Classifier", layout="wide")
st.title("AI-Based Semiconductor Wafer Defect Classification")
st.write(
    "Pick an example wafer or upload your own wafer map. The trained Random Forest predicts "
    "the defect category from 15 numerical features extracted from the map."
)

if not MODEL_FILE.exists() or not SAMPLE_FILE.exists():
    st.error("Model or example files not found. Run the notebook up to Stage 9 first.")
    st.stop()

bundle = load_model()
samples = load_samples()

# ---------- Input ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("1. Choose a wafer")
    source = st.radio("Source", ["Example wafer", "Upload a file"], horizontal=True)
    wafer, true_label = None, None

    if source == "Example wafer":
        names = [s["true_label"] for s in samples]
        chosen = st.selectbox("Example (its real category is shown)", names)
        sample = next(s for s in samples if s["true_label"] == chosen)
        wafer, true_label = sample["waferMap"], sample["true_label"]
    else:
        st.caption("Upload a .npy file, or a .csv/.txt file with comma-separated numbers (0, 1, 2).")
        uploaded = st.file_uploader("Wafer map file", type=["npy", "csv", "txt"])
        if uploaded is not None:
            try:
                wafer = read_uploaded(uploaded)
            except Exception as err:                 # show the problem instead of crashing the page
                st.error(f"Could not read the file: {err}")

    if wafer is not None:
        st.pyplot(draw_wafer(wafer))
        st.caption(f"Wafer grid: {wafer.shape[0]} rows x {wafer.shape[1]} columns")

# ---------- Prediction ----------
with right:
    st.subheader("2. Predict")
    if wafer is None:
        st.info("Choose or upload a wafer on the left first.")
    elif st.button("Predict defect category", type="primary"):
        table = predict(bundle, wafer)
        top_name, top_prob = table.index[0], table.iloc[0]

        st.metric("Predicted category", top_name, f"{top_prob:.0%} confidence")
        if true_label is not None:
            st.write(f"Real category of this example: **{true_label}**")
        st.write(
            "Confidence is the share of the Random Forest's trees that voted for this category. "
            "It is a rough guide, not a guarantee."
        )
        st.bar_chart(table)                          # probability of each category
        st.dataframe(table.rename("probability").map(lambda p: f"{p:.1%}").to_frame())

# ---------- Model performance ----------
st.divider()
st.subheader("Model performance")
st.write(
    "Models were compared on a validation split taken from the Training set. The official Test set "
    "was scored once, at the end. Its scores are lower, mainly for the rare defect classes (see the report)."
)

val_file = REPORTS / "validation_results.csv"
test_file = REPORTS / "test_results.csv"
test_class_file = REPORTS / "test_per_class.csv"

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Four models on validation**")
    if val_file.exists():
        st.dataframe(pd.read_csv(val_file, index_col=0))
    else:
        st.info("validation_results.csv not found.")
with col_b:
    st.markdown("**Chosen model (Random Forest) on the official Test set**")
    if test_file.exists():
        st.dataframe(pd.read_csv(test_file, index_col=0, header=None, names=["score"]))
    if test_class_file.exists():
        st.dataframe(pd.read_csv(test_class_file, index_col=0).round(3))
