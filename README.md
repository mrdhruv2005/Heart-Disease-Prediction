# Heart Disease Risk Prediction

A Streamlit web app that estimates heart-disease risk from clinical measurements using a
trained K-Nearest Neighbours classifier.

> **Not a diagnostic tool.** This is a student machine-learning project. It cannot
> diagnose disease and must never inform an actual medical decision. Consult a qualified
> clinician for anything health-related.

---

## What it does

The app takes standard clinical inputs and returns a risk estimate:

- Age and sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression and slope

These are the features from the well-known UCI heart-disease dataset family, which is
what makes the input schema recognizable to anyone who has worked with this problem.

---

## How inference works

Three artifacts ship together, and all three are required:

| File | Role |
|------|------|
| `knn_heart_model.pkl` | Trained KNN classifier |
| `heart_scaler.pkl` | Fitted `StandardScaler` |
| `heart_columns.pkl` | Column order and schema |

**Why the scaler must ship with the model.** KNN classifies by distance between points.
Without scaling, a feature measured in the hundreds (cholesterol, ~200 mg/dL) dominates the
distance calculation over a feature measured in single digits (ST depression, ~0–4) — the
model would effectively ignore the smaller-scale features regardless of their predictive
value. `StandardScaler` puts every feature on comparable footing.

This makes KNN one of the algorithms where forgetting to scale doesn't degrade performance
subtly, it breaks the model. It also means the *exact* scaler fitted on training data must
be reused at inference — re-fitting on new input would produce different scaling and
therefore wrong predictions.

**Why the column order matters.** The scaler and model both expect features positionally.
`heart_columns.pkl` preserves the training-time order so a form submission is assembled
into the same layout the model was fitted on.

---

## Running locally

```bash
git clone https://github.com/mrdhruv2005/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Repository structure

```
├── app.py                 # Streamlit UI + inference
├── knn_heart_model.pkl    # Trained KNN classifier
├── heart_scaler.pkl       # Fitted StandardScaler
├── heart_columns.pkl      # Feature schema and order
├── requirements.txt
└── LICENSE                # MIT
```

---

## Limitations

**No training code in this repository.** The model artifacts are committed but the
notebook that produced them is not, so the choice of `k`, the train/test split, and the
achieved metrics are not reproducible from what is here. Adding the training notebook is
the most valuable improvement this project could receive.

**No reported metrics.** Accuracy, precision, recall, and ROC-AUC are unknown from this
repo. For a medical screening task, **recall matters far more than accuracy** — a false
negative means telling someone at risk that they are fine. Any honest evaluation of this
model would lead with recall and a confusion matrix, not accuracy.

**KNN has real drawbacks here.** It stores the entire training set, so inference cost
grows with data size; it has no interpretable notion of feature importance, so it cannot
explain *why* a prediction was made — a serious shortcoming in a clinical context; and it
degrades in higher dimensions. Logistic regression would be more interpretable and a
gradient-boosted model likely more accurate. Comparing against both would strengthen the
project considerably.

**Dataset constraints.** The UCI heart-disease data is small (a few hundred rows),
collected decades ago from specific populations, and may not generalize across
demographics.

---

## Tech stack

Python · scikit-learn (KNN, `StandardScaler`) · Pandas · NumPy · Streamlit · joblib

## License

MIT
