# AI-Based Semiconductor Wafer Defect Classification

An academic machine learning project (ECE, 3rd year) that classifies defect
patterns on semiconductor wafer maps using the **WM-811K** dataset.

A wafer map is a grid of pixels showing which chips on a wafer passed or
failed. The system extracts numerical features from each wafer map and
predicts its defect category with classical machine learning
(Decision Tree, Random Forest, SVM, Logistic Regression). No deep learning.

## Project structure

```
AIML_Project/
├── data/
│   ├── raw/              # original dataset (LSWMD.pkl) - NOT committed to git
│   └── processed/        # cleaned or sampled data created by the notebook
├── notebooks/            # Jupyter notebook(s), step by step
├── src/                  # reusable Python code
│   ├── preprocess.py     # cleaning and label handling
│   ├── features.py       # wafer map -> numerical features
│   ├── train.py          # train and compare models
│   └── evaluate.py       # metrics, confusion matrices, plots
├── models/               # saved model files (.joblib)
├── reports/
│   └── figures/          # plots and images for the report
├── app/
│   └── streamlit_app.py  # demo application
├── idea.md               # project brief
├── requirements.txt      # Python packages
└── README.md
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

1. Put `LSWMD.pkl` in `data/raw/`.
2. Open the notebook in `notebooks/` and run the cells in order.
3. Demo app (after a model is saved in `models/`):

```
streamlit run app/streamlit_app.py
```

## Status

- [ ] Data loading and inspection
- [ ] Visualisation
- [ ] Preprocessing
- [ ] Feature extraction
- [ ] Model training and comparison
- [ ] Saved model
- [ ] Demo app
