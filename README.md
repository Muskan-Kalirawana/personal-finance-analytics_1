# 💰 Personal Finance Analytics & Prediction System

A complete end-to-end Data Science project with:
- Synthetic dataset (2,000 users, 28 features)
- SQLite database + SQL queries
- EDA with Pandas
- Visualizations (Matplotlib + Seaborn)
- ML Models (Classification + Regression)
- Interactive Streamlit Dashboard

---

## 🚀 SETUP & RUN (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate data & train models
```bash
python generate_data.py
```
This creates the `data/` and `models/` folders with all CSVs, SQLite DB, and trained ML models.

### Step 3 — Launch the dashboard
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## 📁 Project Structure
```
finance_project/
├── generate_data.py        ← Data pipeline + ML training
├── app.py                  ← Streamlit dashboard (6 pages)
├── requirements.txt        ← Python dependencies
├── README.md
├── data/                   ← Auto-generated after Step 2
│   ├── personal_finance_data.csv
│   ├── personal_finance.db
│   ├── predictions.csv
│   ├── feature_importance.csv
│   ├── monthly_trend.csv
│   ├── kpi_summary.json
│   └── model_meta.json
└── models/                 ← Auto-generated after Step 2
    ├── best_classifier.pkl
    ├── best_regressor.pkl
    └── label_encoder.pkl
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, income distribution, risk pie chart, monthly trend |
| 📊 EDA & Insights | Distributions, correlations heatmap, segmentation analysis |
| 💸 Expense Analysis | Expense breakdown, category comparison, occupation vs income |
| 🗄️ SQL Explorer | Live SQL query editor with 7 preset queries + auto-chart |
| 🤖 ML Models | Model comparison, confusion matrix, feature importance |
| 🔮 Live Predictor | Enter your own data → get risk prediction + savings forecast |

---

## 🤖 ML Models

### Classification — Savings Risk (High / Medium / Low)
- Logistic Regression ← **Best (94%+ accuracy)**
- Decision Tree
- Random Forest
- K-Nearest Neighbors

### Regression — Monthly Savings Amount (₹)
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

## 🛠️ Tech Stack
`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn` `SQLite` `Streamlit` `Joblib`
