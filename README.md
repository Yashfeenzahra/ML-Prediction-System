# 🚢 Titanic Survival Prediction System

Week 3 internship task — an end-to-end Machine Learning project covering data
preprocessing, model training/comparison, evaluation, and a Streamlit app that
predicts whether a passenger would have survived the Titanic.

## Live demo
> Add your deployed Streamlit link here once deployed, e.g.
> `https://your-app-name.streamlit.app`

## What this project does

1. Cleans and preprocesses the Titanic dataset (missing values, duplicates,
   encoding, scaling).
2. Trains and compares three classifiers: Logistic Regression, Decision Tree,
   and K-Nearest Neighbors.
3. Evaluates each model with accuracy, precision, recall, F1-score, and
   confusion matrices.
4. Saves the best-performing model with `joblib`.
5. Serves everything through an interactive Streamlit app where you can enter
   a passenger's details and get a survival prediction.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.814 | 0.769 | 0.781 | 0.775 |
| Decision Tree | 0.769 | 0.818 | 0.563 | 0.667 |
| KNN | 0.776 | 0.796 | 0.609 | 0.690 |

Logistic Regression came out on top on this split and is the model used by the
app for predictions.

## Project structure

```
Week3-ML-Prediction-System/
│
├── data/
│   └── Titanic-Dataset.csv
├── notebooks/
│   └── training.ipynb          # full data prep + training + evaluation
├── models/
│   ├── trained_model.pkl       # best model (Logistic Regression)
│   ├── scaler.pkl              # StandardScaler used at train time
│   ├── le_sex.pkl              # LabelEncoder for Sex
│   ├── le_emb.pkl              # LabelEncoder for Embarked
│   ├── feature_columns.pkl     # column order expected by the model
│   └── results.pkl             # metrics for each trained model
├── screenshots/
├── app.py                      # Streamlit app
├── requirements.txt
└── README.md
```

## Running it locally

```bash
# 1. clone the repo
git clone https://github.com/<your-username>/Week3-ML-Prediction-System.git
cd Week3-ML-Prediction-System

# 2. create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Re-training the model

Open `notebooks/training.ipynb` in Jupyter or Google Colab and run all cells.
It will regenerate the files in `models/`. The notebook auto-detects where the
data lives: if you upload `Titanic-Dataset.csv` to Colab (so it sits at
`/content/Titanic-Dataset.csv`), it'll use that automatically; otherwise it
falls back to `../data/Titanic-Dataset.csv`, which is where it lives inside
this repo.

## Dataset

Titanic passenger data (891 rows, 12 columns) — the classic dataset used for
binary classification practice. Target column: `Survived` (0 = did not
survive, 1 = survived).

## Tech stack

Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit ·
Jupyter Notebook

## Notes / possible improvements

- Try ensemble models (Random Forest, Gradient Boosting) for a stronger
  baseline.
- Hyperparameter tuning with `GridSearchCV`.
- Extract a "title" feature (Mr/Mrs/Miss/Master) from the `Name` column
  instead of dropping it entirely.
- Deploy on Streamlit Community Cloud (see below) and link it here.

## Author

Built as part of a Week 3 Machine Learning internship task.
