# =============================================================================
#  PERSONAL FINANCE ANALYTICS & PREDICTION SYSTEM
#  Step 1: Data Generation + SQL + ML Pipeline
#  Run this FIRST before launching the Streamlit app
#  Command: python generate_data.py
# =============================================================================

import os, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error,
    ConfusionMatrixDisplay
)
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("  PERSONAL FINANCE ANALYTICS & PREDICTION SYSTEM")
print("  Generating Data + Training Models...")
print("=" * 60)

# =============================================================================
# 1. GENERATE DATASET
# =============================================================================
N = 2000
age        = np.random.randint(22, 65, N)
gender     = np.random.choice(["Male", "Female", "Other"], N, p=[0.48, 0.48, 0.04])
education  = np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], N, p=[0.20, 0.45, 0.25, 0.10])
occupation = np.random.choice(["Salaried", "Self-Employed", "Freelancer", "Business Owner", "Student"], N, p=[0.45, 0.20, 0.15, 0.12, 0.08])
city_tier  = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], N, p=[0.35, 0.40, 0.25])

base_income = {"Salaried": 55000, "Self-Employed": 48000, "Freelancer": 40000, "Business Owner": 72000, "Student": 15000}
edu_bonus   = {"High School": 0, "Bachelor's": 8000, "Master's": 15000, "PhD": 22000}

monthly_income = np.array([
    int(base_income[occupation[i]] + edu_bonus[education[i]] + np.random.normal(0, 8000))
    for i in range(N)
]).clip(10000, 200000)

housing_rent      = (monthly_income * np.random.uniform(0.15, 0.35, N)).astype(int)
groceries         = np.random.randint(2000, 12000, N)
utilities         = np.random.randint(500,  4000,  N)
transport         = np.random.randint(500,  8000,  N)
dining_out        = np.random.randint(500,  10000, N)
entertainment     = np.random.randint(200,  6000,  N)
healthcare        = np.random.randint(200,  5000,  N)
education_expense = np.where(age < 30, np.random.randint(0, 8000, N), np.random.randint(0, 2000, N))
shopping          = np.random.randint(500,  15000, N)
investments       = (monthly_income * np.random.uniform(0.0, 0.25, N)).astype(int)
loan_emi          = np.where(np.random.rand(N) > 0.4, (monthly_income * np.random.uniform(0.05, 0.30, N)).astype(int), 0)
miscellaneous     = np.random.randint(200,  5000,  N)

total_expenses   = (housing_rent + groceries + utilities + transport + dining_out +
                    entertainment + healthcare + education_expense + shopping + loan_emi + miscellaneous)
monthly_savings  = (monthly_income - total_expenses - investments).clip(0)
savings_rate     = (monthly_savings / monthly_income.clip(1) * 100).round(2)
expense_ratio    = (total_expenses   / monthly_income.clip(1) * 100).round(2)
investment_ratio = (investments      / monthly_income.clip(1) * 100).round(2)
has_loan         = (loan_emi > 0).astype(int)

credit_score = (
    500 + savings_rate * 3 - expense_ratio * 0.5 +
    monthly_income / 1000 - has_loan * 30 + np.random.normal(0, 20, N)
).clip(300, 900).astype(int)

savings_risk = pd.cut(savings_rate, bins=[-np.inf, 10, 20, np.inf],
                      labels=["High Risk", "Medium Risk", "Low Risk"]).astype(str)

months = np.random.choice(pd.date_range("2022-01-01", "2024-12-01", freq="MS"), N)

df = pd.DataFrame({
    "user_id": [f"USR{str(i+1).zfill(4)}" for i in range(N)],
    "month": months, "age": age, "gender": gender,
    "education": education, "occupation": occupation, "city_tier": city_tier,
    "monthly_income": monthly_income, "housing_rent": housing_rent,
    "groceries": groceries, "utilities": utilities, "transport": transport,
    "dining_out": dining_out, "entertainment": entertainment,
    "healthcare": healthcare, "education_expense": education_expense,
    "shopping": shopping, "investments": investments, "loan_emi": loan_emi,
    "miscellaneous": miscellaneous, "total_expenses": total_expenses,
    "monthly_savings": monthly_savings, "savings_rate": savings_rate,
    "expense_ratio": expense_ratio, "investment_ratio": investment_ratio,
    "has_loan": has_loan, "credit_score": credit_score, "savings_risk": savings_risk,
})
df["age_group"] = pd.cut(df["age"], bins=[21,30,40,50,65], labels=["22-30","31-40","41-50","51-65"]).astype(str)
df["month_str"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)

df.to_csv("data/personal_finance_data.csv", index=False)
print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} cols saved.")

# =============================================================================
# 2. SQLITE
# =============================================================================
conn = sqlite3.connect("data/personal_finance.db")
df.to_sql("finance", conn, if_exists="replace", index=False)
conn.close()
print("  SQLite DB saved.")

# =============================================================================
# 3. ML MODELS
# =============================================================================
FEATURES = ["age","gender","education","occupation","city_tier","monthly_income",
            "housing_rent","groceries","utilities","transport","dining_out",
            "entertainment","healthcare","education_expense","shopping","investments",
            "loan_emi","miscellaneous","total_expenses","expense_ratio",
            "investment_ratio","has_loan","credit_score"]
cat_cols = ["gender","education","occupation","city_tier"]
num_cols = [c for c in FEATURES if c not in cat_cols]

le = LabelEncoder()
df["savings_risk_enc"] = le.fit_transform(df["savings_risk"])

X = df[FEATURES]
y_clf = df["savings_risk_enc"]
y_reg = df["monthly_savings"]

X_train, X_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
_, _, yr_train, yr_test             = train_test_split(X, y_reg, test_size=0.2, random_state=42)

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

# ── Classification ────────────────────────────────────────────────────────────
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
}
clf_results = {}
best_clf, best_clf_name, best_acc = None, "", 0
for name, clf in classifiers.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, yc_train)
    acc = accuracy_score(yc_test, pipe.predict(X_test))
    cv  = cross_val_score(pipe, X, y_clf, cv=StratifiedKFold(5), scoring="accuracy")
    clf_results[name] = {"accuracy": round(acc,4), "cv_mean": round(cv.mean(),4), "cv_std": round(cv.std(),4)}
    if acc > best_acc:
        best_acc, best_clf_name, best_clf = acc, name, pipe
    print(f"  [CLF] {name:<25} Acc: {acc:.4f}  CV: {cv.mean():.4f}±{cv.std():.4f}")

y_pred_clf = best_clf.predict(X_test)
report_dict = classification_report(yc_test, y_pred_clf, target_names=le.classes_, output_dict=True)

# ── Regression ────────────────────────────────────────────────────────────────
regressors = {
    "Ridge Regression":        Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=150, random_state=42),
}
reg_results = {}
best_reg, best_reg_name, best_r2 = None, "", -np.inf
for name, reg in regressors.items():
    pipe = Pipeline([("pre", pre), ("reg", reg)])
    pipe.fit(X_train, yr_train)
    yp   = pipe.predict(X_test)
    r2   = r2_score(yr_test, yp)
    mae  = mean_absolute_error(yr_test, yp)
    rmse = np.sqrt(mean_squared_error(yr_test, yp))
    reg_results[name] = {"R2": round(r2,4), "MAE": round(mae,2), "RMSE": round(rmse,2), "pred": yp.tolist()}
    if r2 > best_r2:
        best_r2, best_reg_name, best_reg = r2, name, pipe
    print(f"  [REG] {name:<28} R²: {r2:.4f}  MAE: {mae:,.0f}  RMSE: {rmse:,.0f}")

y_pred_reg = np.array(reg_results[best_reg_name]["pred"])

# ── Feature importance ────────────────────────────────────────────────────────
ohe_feats  = best_clf.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols).tolist()
feat_names = num_cols + ohe_feats
if hasattr(best_clf.named_steps["clf"], "feature_importances_"):
    imp = best_clf.named_steps["clf"].feature_importances_
else:
    imp = np.abs(best_clf.named_steps["clf"].coef_).mean(axis=0)
feat_imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False).head(15)
feat_imp_df.to_csv("data/feature_importance.csv", index=False)

# ── Save prediction results ───────────────────────────────────────────────────
pred_df = X_test.copy().reset_index(drop=True)
pred_df["actual_risk"]        = le.inverse_transform(yc_test.values)
pred_df["predicted_risk"]     = le.inverse_transform(y_pred_clf)
pred_df["actual_savings"]     = yr_test.values
pred_df["predicted_savings"]  = y_pred_reg.round(0)
pred_df["correct"]            = (pred_df["actual_risk"] == pred_df["predicted_risk"]).astype(int)
pred_df.to_csv("data/predictions.csv", index=False)

# ── Save model metadata ───────────────────────────────────────────────────────
import json
meta = {
    "clf_results":    clf_results,
    "reg_results":    {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in reg_results.items()},
    "best_clf":       best_clf_name,
    "best_clf_acc":   best_acc,
    "best_reg":       best_reg_name,
    "best_reg_r2":    best_r2,
    "classes":        le.classes_.tolist(),
    "report":         report_dict,
    "cm":             confusion_matrix(yc_test, y_pred_clf).tolist(),
}
with open("data/model_meta.json", "w") as f:
    json.dump(meta, f)

joblib.dump(best_clf, "models/best_classifier.pkl")
joblib.dump(best_reg, "models/best_regressor.pkl")
joblib.dump(le,       "models/label_encoder.pkl")
print("  Models saved.")

# =============================================================================
# 4. MONTHLY TREND
# =============================================================================
trend = df.groupby("month_str")[["monthly_income","monthly_savings","total_expenses","investments"]].mean().reset_index()
trend.to_csv("data/monthly_trend.csv", index=False)

# =============================================================================
# 5. KPI SUMMARY
# =============================================================================
kpi = {
    "total_users":        int(len(df)),
    "avg_income":         round(float(df["monthly_income"].mean()), 2),
    "avg_savings":        round(float(df["monthly_savings"].mean()), 2),
    "avg_savings_rate":   round(float(df["savings_rate"].mean()), 2),
    "avg_credit_score":   round(float(df["credit_score"].mean()), 2),
    "avg_expense_ratio":  round(float(df["expense_ratio"].mean()), 2),
    "high_risk_pct":      round(float((df["savings_risk"]=="High Risk").mean()*100), 2),
    "medium_risk_pct":    round(float((df["savings_risk"]=="Medium Risk").mean()*100), 2),
    "low_risk_pct":       round(float((df["savings_risk"]=="Low Risk").mean()*100), 2),
    "loan_holders_pct":   round(float(df["has_loan"].mean()*100), 2),
}
with open("data/kpi_summary.json", "w") as f:
    json.dump(kpi, f)

print("\n  All data files ready in ./data/")
print("  Now run:  streamlit run app.py\n")
