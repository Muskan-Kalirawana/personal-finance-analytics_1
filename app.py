"""
=============================================================================
  PERSONAL FINANCE ANALYTICS & PREDICTION SYSTEM
  Streamlit Dashboard — app.py
  Run: streamlit run app.py
=============================================================================
"""

import json, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
import joblib
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────────────────────────────────────
CLR_GREEN  = "#2ecc71"
CLR_ORANGE = "#f39c12"
CLR_RED    = "#e74c3c"
CLR_BLUE   = "#3498db"
CLR_PURPLE = "#9b59b6"
RISK_COLORS = {"Low Risk": CLR_GREEN, "Medium Risk": CLR_ORANGE, "High Risk": CLR_RED}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* KPI Card */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 4px solid #3498db;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-label { color: #aab4c8; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .kpi-value { color: #ffffff; font-size: 1.65rem; font-weight: 700; line-height: 1.1; }
    .kpi-sub   { color: #6c7a9c; font-size: 0.75rem; margin-top: 3px; }

    /* Risk Badge */
    .badge-high   { background:#e74c3c22; color:#e74c3c; border:1px solid #e74c3c55; border-radius:6px; padding:2px 10px; font-size:0.82rem; font-weight:600; }
    .badge-medium { background:#f39c1222; color:#f39c12; border:1px solid #f39c1255; border-radius:6px; padding:2px 10px; font-size:0.82rem; font-weight:600; }
    .badge-low    { background:#2ecc7122; color:#2ecc71; border:1px solid #2ecc7155; border-radius:6px; padding:2px 10px; font-size:0.82rem; font-weight:600; }

    /* Section title */
    .section-title { font-size:1.15rem; font-weight:700; color:#e0e6f0; margin-bottom:10px; padding-bottom:6px; border-bottom:2px solid #252840; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b27 100%); }
    [data-testid="stSidebar"] .stRadio label { color: #c9d1d9; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background:#161b27; border-radius:8px; gap:4px; }
    .stTabs [data-baseweb="tab"]      { color:#8b949e; font-weight:600; border-radius:6px; }
    .stTabs [aria-selected="true"]    { background:#21262d; color:#58a6ff; }
    
    /* Metric override */
    [data-testid="metric-container"] { background:#1e2130; border-radius:10px; padding:12px; border:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df    = pd.read_csv("data/personal_finance_data.csv")
    trend = pd.read_csv("data/monthly_trend.csv")
    preds = pd.read_csv("data/predictions.csv")
    fi    = pd.read_csv("data/feature_importance.csv")
    with open("data/kpi_summary.json")  as f: kpi  = json.load(f)
    with open("data/model_meta.json")   as f: meta = json.load(f)
    df["savings_risk"] = pd.Categorical(df["savings_risk"],
        categories=["Low Risk","Medium Risk","High Risk"], ordered=True)
    return df, trend, preds, fi, kpi, meta

@st.cache_resource
def load_models():
    clf = joblib.load("models/best_classifier.pkl")
    reg = joblib.load("models/best_regressor.pkl")
    le  = joblib.load("models/label_encoder.pkl")
    return clf, reg, le

try:
    df, trend, preds, feat_imp, kpi, meta = load_data()
    clf_model, reg_model, label_enc       = load_models()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.error(f"⚠️  Data not found. Please run `python generate_data.py` first.\n\nError: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVIGATION + GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Finance Analytics")
    st.markdown("---")
    page = st.radio("📌 **Navigation**", [
        "🏠 Overview",
        "📊 EDA & Insights",
        "💸 Expense Analysis",
        "🗄️ SQL Explorer",
        "🤖 ML Models",
        "🔮 Live Predictor",
    ])
    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")
    sel_occ   = st.multiselect("Occupation",  df["occupation"].unique().tolist(),  default=df["occupation"].unique().tolist())
    sel_city  = st.multiselect("City Tier",   df["city_tier"].unique().tolist(),   default=df["city_tier"].unique().tolist())
    sel_risk  = st.multiselect("Savings Risk",["Low Risk","Medium Risk","High Risk"], default=["Low Risk","Medium Risk","High Risk"])
    inc_range = st.slider("Income Range (₹)", int(df["monthly_income"].min()), int(df["monthly_income"].max()),
                          (int(df["monthly_income"].min()), int(df["monthly_income"].max())), step=5000)
    st.markdown("---")
    st.markdown("<small style='color:#555'>👤 2,000 synthetic users<br>📅 Jan 2022 – Dec 2024</small>", unsafe_allow_html=True)

# Apply global filters
dff = df[
    df["occupation"].isin(sel_occ) &
    df["city_tier"].isin(sel_city) &
    df["savings_risk"].isin(sel_risk) &
    df["monthly_income"].between(inc_range[0], inc_range[1])
].copy()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: KPI card HTML
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", border_color=CLR_BLUE):
    return f"""
    <div class="kpi-card" style="border-left-color:{border_color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

def fmt_inr(v): return f"₹{v:,.0f}"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: matplotlib dark figure
# ─────────────────────────────────────────────────────────────────────────────
def dark_fig(w=10, h=5, nrows=1, ncols=1, **kwargs):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    fig.patch.set_facecolor("#0f1117")
    def _style(a):
        a.set_facecolor("#161b27")
        a.tick_params(colors="#8b949e", labelsize=9)
        a.xaxis.label.set_color("#8b949e")
        a.yaxis.label.set_color("#8b949e")
        a.title.set_color("#e0e6f0")
        for spine in a.spines.values():
            spine.set_edgecolor("#30363d")
    if nrows == 1 and ncols == 1:
        _style(ax)
    else:
        for a in (ax.flat if hasattr(ax, "flat") else [ax]):
            _style(a)
    return fig, ax


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
if page == "🏠 Overview":
    st.markdown("# 🏦 Personal Finance Analytics Dashboard")
    st.markdown(f"*Showing **{len(dff):,}** users based on current filters*")
    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "Total Users",        f"{len(dff):,}",             "Filtered sample",        CLR_BLUE),
        (c2, "Avg Monthly Income", fmt_inr(dff['monthly_income'].mean()), "Per user",      CLR_GREEN),
        (c3, "Avg Monthly Savings",fmt_inr(dff['monthly_savings'].mean()),"Per user",      CLR_ORANGE),
        (c4, "Avg Savings Rate",   f"{dff['savings_rate'].mean():.1f}%",  "Of income",    CLR_PURPLE),
        (c5, "Avg Credit Score",   f"{dff['credit_score'].mean():.0f}",   "Out of 900",   CLR_RED),
        (c6, "High Risk Users",    f"{(dff['savings_risk']=='High Risk').mean()*100:.1f}%","Need attention", CLR_RED),
    ]
    for col, lbl, val, sub, clr in cards:
        col.markdown(kpi_card(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">💵 Income Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        ax.hist(dff["monthly_income"], bins=40, color=CLR_BLUE, edgecolor="#0f1117", alpha=0.85)
        ax.axvline(dff["monthly_income"].median(), color=CLR_ORANGE, linestyle="--", linewidth=2,
                   label=f"Median: {fmt_inr(dff['monthly_income'].median())}")
        ax.axvline(dff["monthly_income"].mean(), color=CLR_GREEN, linestyle="--", linewidth=2,
                   label=f"Mean: {fmt_inr(dff['monthly_income'].mean())}")
        ax.set_xlabel("Monthly Income (₹)"); ax.set_ylabel("Count")
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-title">🎯 Savings Risk Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        risk_counts = dff["savings_risk"].value_counts()
        wedges, texts, autotexts = ax.pie(
            risk_counts, labels=risk_counts.index, autopct="%1.1f%%",
            colors=[RISK_COLORS.get(r, CLR_BLUE) for r in risk_counts.index],
            startangle=140, wedgeprops={"edgecolor": "#0f1117", "linewidth": 2},
        )
        for t in texts:     t.set_color("#c9d1d9")
        for a in autotexts: a.set_color("#ffffff"); a.set_fontweight("bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    col3, col4 = st.columns([1.2, 0.8])

    with col3:
        st.markdown('<div class="section-title">📅 Income vs Savings Monthly Trend</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(9, 4)
        t = trend.sort_values("month_str")
        ax.plot(t["month_str"], t["monthly_income"],   marker="o", color=CLR_BLUE,   linewidth=2, label="Avg Income",   markersize=4)
        ax.plot(t["month_str"], t["total_expenses"],   marker="s", color=CLR_RED,    linewidth=2, label="Avg Expenses", markersize=4)
        ax.plot(t["month_str"], t["monthly_savings"],  marker="^", color=CLR_GREEN,  linewidth=2, label="Avg Savings",  markersize=4)
        ax.fill_between(t["month_str"], t["monthly_savings"], alpha=0.12, color=CLR_GREEN)
        ax.set_xlabel("Month"); ax.set_ylabel("Amount (₹)")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown('<div class="section-title">👔 Avg Income by Occupation</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 4)
        occ_inc = dff.groupby("occupation")["monthly_income"].mean().sort_values()
        colors  = sns.color_palette("Blues", len(occ_inc))
        bars = ax.barh(occ_inc.index, occ_inc.values, color=colors, edgecolor="#0f1117")
        for bar, val in zip(bars, occ_inc.values):
            ax.text(val + 500, bar.get_y() + bar.get_height()/2, fmt_inr(val), va="center", fontsize=8, color="#c9d1d9")
        ax.set_xlabel("Monthly Income (₹)")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # ── Row 3: scatter ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🫧 Income vs Savings (colored by Savings Risk)</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(14, 5)
    for risk, grp in dff.groupby("savings_risk", observed=True):
        ax.scatter(grp["monthly_income"], grp["monthly_savings"],
                   color=RISK_COLORS.get(str(risk), CLR_BLUE), alpha=0.45, s=18, label=str(risk))
    ax.set_xlabel("Monthly Income (₹)"); ax.set_ylabel("Monthly Savings (₹)")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# =============================================================================
# PAGE 2 — EDA & INSIGHTS
# =============================================================================
elif page == "📊 EDA & Insights":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown(f"*Filtered dataset: {len(dff):,} users*")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 Distributions", "🔗 Correlations", "👥 Segmentation"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Shape**")
            st.info(f"Rows: {dff.shape[0]:,}  |  Columns: {dff.shape[1]}")
            st.markdown("**Data Types**")
            st.dataframe(pd.DataFrame({"dtype": dff.dtypes.astype(str)}), use_container_width=True)
        with c2:
            st.markdown("**Descriptive Statistics**")
            num_summary = dff[["monthly_income","total_expenses","monthly_savings",
                                "savings_rate","expense_ratio","credit_score"]].describe().round(2)
            st.dataframe(num_summary, use_container_width=True)
            st.markdown("**Savings Risk Distribution**")
            rc = dff["savings_risk"].value_counts().reset_index()
            rc.columns = ["Risk Level", "Count"]; rc["Percentage"] = (rc["Count"] / len(dff) * 100).round(2)
            st.dataframe(rc, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            feat = st.selectbox("Select Feature", ["monthly_income","monthly_savings","savings_rate",
                                                    "expense_ratio","credit_score","investment_ratio","total_expenses"])
            fig, ax = dark_fig(7, 4)
            ax.hist(dff[feat].dropna(), bins=40, color=CLR_BLUE, edgecolor="#0f1117", alpha=0.85)
            ax.axvline(dff[feat].mean(), color=CLR_ORANGE, linestyle="--", linewidth=2, label=f"Mean: {dff[feat].mean():.2f}")
            ax.axvline(dff[feat].median(), color=CLR_GREEN, linestyle="--", linewidth=2, label=f"Median: {dff[feat].median():.2f}")
            ax.set_title(f"Distribution of {feat}"); ax.set_xlabel(feat); ax.set_ylabel("Count")
            ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            fig, ax = dark_fig(7, 4)
            risk_order = ["Low Risk", "Medium Risk", "High Risk"]
            data_violin = [dff[dff["savings_risk"] == r][feat].dropna() for r in risk_order]
            parts = ax.violinplot(data_violin, positions=[1,2,3], showmedians=True, showmeans=False)
            for pc, clr in zip(parts["bodies"], [CLR_GREEN, CLR_ORANGE, CLR_RED]):
                pc.set_facecolor(clr); pc.set_alpha(0.6)
            parts["cmedians"].set_color("white"); parts["cbars"].set_color("#30363d")
            parts["cmins"].set_color("#30363d");  parts["cmaxes"].set_color("#30363d")
            ax.set_xticks([1,2,3]); ax.set_xticklabels(risk_order)
            ax.set_title(f"{feat} by Savings Risk"); ax.set_ylabel(feat)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        num_cols_corr = ["monthly_income","total_expenses","monthly_savings","savings_rate",
                         "investment_ratio","credit_score","expense_ratio","loan_emi","housing_rent"]
        corr = dff[num_cols_corr].corr()
        fig, ax = dark_fig(10, 7)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.4, ax=ax, annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Heatmap", pad=15)
        ax.tick_params(axis="x", rotation=40)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("**Top Correlations with Monthly Savings**")
        corr_savings = dff[num_cols_corr].corr()["monthly_savings"].drop("monthly_savings").sort_values(key=abs, ascending=False)
        st.dataframe(corr_savings.rename("Correlation").reset_index(), use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = dark_fig(7, 5)
            risk_edu = pd.crosstab(dff["education"], dff["savings_risk"], normalize="index") * 100
            risk_edu = risk_edu.reindex(columns=["Low Risk","Medium Risk","High Risk"], fill_value=0)
            risk_edu.plot(kind="bar", stacked=True, ax=ax,
                color=[CLR_GREEN, CLR_ORANGE, CLR_RED], edgecolor="#0f1117", width=0.65)
            ax.set_title("Savings Risk by Education Level"); ax.set_xlabel("Education"); ax.set_ylabel("%")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            fig, ax = dark_fig(7, 5)
            hm = dff.groupby(["occupation","city_tier"])["savings_rate"].mean().unstack()
            sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlGn", linewidths=0.4,
                        ax=ax, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
            ax.set_title("Avg Savings Rate % — Occupation × City Tier")
            ax.tick_params(axis="x", rotation=15)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        fig, ax = dark_fig(12, 4)
        age_sav = dff.groupby("age_group")["savings_rate"].mean().reindex(["22-30","31-40","41-50","51-65"])
        bars = ax.bar(age_sav.index, age_sav.values, color=[CLR_BLUE, CLR_GREEN, CLR_ORANGE, CLR_PURPLE], edgecolor="#0f1117", width=0.5)
        for bar, val in zip(bars, age_sav.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}%", ha="center", color="#c9d1d9", fontweight="bold")
        ax.set_title("Avg Savings Rate by Age Group"); ax.set_xlabel("Age Group"); ax.set_ylabel("Savings Rate %")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# =============================================================================
# PAGE 3 — EXPENSE ANALYSIS
# =============================================================================
elif page == "💸 Expense Analysis":
    st.markdown("# 💸 Expense Deep-Dive")
    st.markdown(f"*Filtered dataset: {len(dff):,} users*")
    st.markdown("---")

    EXPENSE_COLS = ["housing_rent","groceries","shopping","dining_out",
                    "entertainment","transport","loan_emi","healthcare",
                    "utilities","education_expense","miscellaneous"]

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Avg Total Expenses", fmt_inr(dff["total_expenses"].mean()), "Monthly", CLR_RED), unsafe_allow_html=True)
    c2.markdown(kpi_card("Avg Expense Ratio",  f"{dff['expense_ratio'].mean():.1f}%", "Of income", CLR_ORANGE), unsafe_allow_html=True)
    c3.markdown(kpi_card("Loan Holders",        f"{dff['has_loan'].mean()*100:.1f}%", "Have EMI",  CLR_PURPLE), unsafe_allow_html=True)
    c4.markdown(kpi_card("Avg Investments",     fmt_inr(dff["investments"].mean()),   "Monthly",   CLR_GREEN), unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown('<div class="section-title">📊 Avg Spend by Expense Category</div>', unsafe_allow_html=True)
        exp_avg = {c: dff[c].mean() for c in EXPENSE_COLS}
        exp_sorted = dict(sorted(exp_avg.items(), key=lambda x: x[1], reverse=True))
        fig, ax = dark_fig(8, 6)
        colors_bar = sns.color_palette("Blues_r", len(exp_sorted))
        bars = ax.barh(list(exp_sorted.keys()), list(exp_sorted.values()), color=colors_bar, edgecolor="#0f1117")
        for bar, val in zip(bars, exp_sorted.values()):
            ax.text(val + 100, bar.get_y() + bar.get_height()/2, fmt_inr(val), va="center", fontsize=8, color="#c9d1d9")
        ax.set_xlabel("Avg Monthly Amount (₹)")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-title">🍕 Expense Share (Avg User)</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 6)
        vals   = list(exp_sorted.values())
        labels = [k.replace("_", " ").title() for k in exp_sorted.keys()]
        colors_pie = sns.color_palette("tab10", len(vals))
        wedges, texts, autotexts = ax.pie(vals, labels=labels, autopct="%1.1f%%",
            colors=colors_pie, startangle=140,
            wedgeprops={"edgecolor": "#0f1117", "linewidth": 1.5},
            textprops={"fontsize": 8})
        for t in texts:     t.set_color("#c9d1d9")
        for a in autotexts: a.set_color("#ffffff"); a.set_fontsize(7)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # ── Expense by Risk ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚖️ Expense Category by Savings Risk Level</div>', unsafe_allow_html=True)
    risk_order3 = ["Low Risk", "Medium Risk", "High Risk"]
    exp_risk = dff.groupby("savings_risk", observed=False)[EXPENSE_COLS].mean().reindex(risk_order3)
    fig, ax = dark_fig(14, 5)
    x   = np.arange(len(EXPENSE_COLS))
    w   = 0.25
    clrs = [CLR_GREEN, CLR_ORANGE, CLR_RED]
    for i, (risk, row) in enumerate(exp_risk.iterrows()):
        ax.bar(x + i*w, row.values, w, label=risk, color=clrs[i], edgecolor="#0f1117", alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([c.replace("_"," ").title() for c in EXPENSE_COLS], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg Monthly Spend (₹)")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # ── Grouped: Income/Expense/Savings by Occupation ──────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">👔 Income / Expenses / Savings by Occupation</div>', unsafe_allow_html=True)
        occ_sum = dff.groupby("occupation")[["monthly_income","total_expenses","monthly_savings"]].mean()
        fig, ax = dark_fig(8, 5)
        x2 = np.arange(len(occ_sum)); w2 = 0.25
        ax.bar(x2 - w2, occ_sum["monthly_income"],   w2, label="Income",   color=CLR_BLUE,   edgecolor="#0f1117")
        ax.bar(x2,      occ_sum["total_expenses"],   w2, label="Expenses", color=CLR_RED,    edgecolor="#0f1117")
        ax.bar(x2 + w2, occ_sum["monthly_savings"],  w2, label="Savings",  color=CLR_GREEN,  edgecolor="#0f1117")
        ax.set_xticks(x2); ax.set_xticklabels(occ_sum.index, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Amount (₹)")
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown('<div class="section-title">📦 Expense Ratio Boxplot by Savings Risk</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 5)
        data_box = [dff[dff["savings_risk"] == r]["expense_ratio"].dropna() for r in risk_order3]
        bp = ax.boxplot(data_box, patch_artist=True, medianprops={"color":"white","linewidth":2})
        for patch, clr in zip(bp["boxes"], [CLR_GREEN, CLR_ORANGE, CLR_RED]):
            patch.set_facecolor(clr); patch.set_alpha(0.6)
        for element in ["whiskers","caps","fliers"]:
            for item in bp[element]: item.set_color("#8b949e")
        ax.set_xticklabels(risk_order3); ax.set_ylabel("Expense Ratio (%)")
        ax.set_title("Expense Ratio by Savings Risk")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("**📋 Raw Expense Summary by Occupation**")
    exp_table = dff.groupby("occupation")[EXPENSE_COLS + ["total_expenses"]].mean().round(0).astype(int).reset_index()
    st.dataframe(exp_table, use_container_width=True)


# =============================================================================
# PAGE 4 — SQL EXPLORER
# =============================================================================
elif page == "🗄️ SQL Explorer":
    st.markdown("# 🗄️ SQL Explorer")
    st.markdown("Directly query the SQLite database. Table name: **`finance`**")
    st.markdown("---")

    PRESET_QUERIES = {
        "-- Select a preset --": "",
        "Avg Income & Savings by Occupation": """SELECT occupation,
       ROUND(AVG(monthly_income), 0)  AS avg_income,
       ROUND(AVG(monthly_savings), 0) AS avg_savings,
       ROUND(AVG(savings_rate), 2)    AS avg_savings_rate_pct
FROM finance
GROUP BY occupation
ORDER BY avg_income DESC;""",
        "Savings Risk Distribution": """SELECT savings_risk,
       COUNT(*) AS total_users,
       ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM finance), 2) AS pct
FROM finance
GROUP BY savings_risk
ORDER BY total_users DESC;""",
        "Top 5 Expense Categories": """SELECT 'Housing Rent' AS category, ROUND(AVG(housing_rent),0) AS avg_spend FROM finance
UNION ALL SELECT 'Groceries',    ROUND(AVG(groceries),0)    FROM finance
UNION ALL SELECT 'Shopping',     ROUND(AVG(shopping),0)     FROM finance
UNION ALL SELECT 'Dining Out',   ROUND(AVG(dining_out),0)   FROM finance
UNION ALL SELECT 'Loan EMI',     ROUND(AVG(loan_emi),0)     FROM finance
UNION ALL SELECT 'Transport',    ROUND(AVG(transport),0)    FROM finance
ORDER BY avg_spend DESC LIMIT 5;""",
        "High Income but High Risk Users": """SELECT user_id, occupation, monthly_income, monthly_savings,
       total_expenses, expense_ratio, savings_risk
FROM finance
WHERE savings_risk = 'High Risk' AND monthly_income > 60000
ORDER BY monthly_income DESC LIMIT 15;""",
        "Credit Score Brackets": """SELECT
  CASE WHEN credit_score < 500 THEN 'Poor (<500)'
       WHEN credit_score BETWEEN 500 AND 649 THEN 'Fair (500-649)'
       WHEN credit_score BETWEEN 650 AND 749 THEN 'Good (650-749)'
       ELSE 'Excellent (750+)' END AS bracket,
  COUNT(*) AS users,
  ROUND(AVG(monthly_savings), 0) AS avg_savings
FROM finance GROUP BY bracket ORDER BY users DESC;""",
        "City Tier Investment Analysis": """SELECT city_tier,
       ROUND(AVG(investment_ratio), 2) AS avg_invest_pct,
       ROUND(AVG(monthly_savings), 0)  AS avg_savings,
       ROUND(AVG(credit_score), 0)     AS avg_credit
FROM finance
GROUP BY city_tier ORDER BY avg_invest_pct DESC;""",
        "Monthly Average Trend": """SELECT month_str,
       ROUND(AVG(monthly_income), 0)  AS avg_income,
       ROUND(AVG(total_expenses), 0)  AS avg_expenses,
       ROUND(AVG(monthly_savings), 0) AS avg_savings
FROM finance
GROUP BY month_str ORDER BY month_str LIMIT 24;""",
    }

    preset = st.selectbox("📌 Preset Queries", list(PRESET_QUERIES.keys()))
    query  = st.text_area("✏️ SQL Query", value=PRESET_QUERIES[preset], height=160,
                          placeholder="SELECT * FROM finance LIMIT 10;")

    col_run, col_dl = st.columns([1, 5])
    run_btn = col_run.button("▶ Run Query", type="primary")

    if run_btn and query.strip():
        try:
            conn   = sqlite3.connect("data/personal_finance.db")
            result = pd.read_sql_query(query, conn)
            conn.close()
            st.success(f"✅ {len(result):,} rows returned")
            st.dataframe(result, use_container_width=True)
            csv = result.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv, "query_result.csv", "text/csv")

            # Auto-chart if 2 cols (categorical + numeric)
            if len(result.columns) == 2:
                c1, c2 = result.columns
                if result[c2].dtype in [np.float64, np.int64]:
                    fig, ax = dark_fig(9, 4)
                    ax.barh(result[c1].astype(str), result[c2],
                            color=sns.color_palette("Blues_r", len(result)), edgecolor="#0f1117")
                    ax.set_title(f"{c2} by {c1}"); ax.set_xlabel(c2)
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        except Exception as e:
            st.error(f"SQL Error: {e}")

    st.markdown("---")
    st.markdown("**📐 Available Columns in `finance` table**")
    st.code(", ".join(df.columns.tolist()), language="text")


# =============================================================================
# PAGE 5 — ML MODELS
# =============================================================================
elif page == "🤖 ML Models":
    st.markdown("# 🤖 Machine Learning Models")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🏷️ Classification (Savings Risk)", "📉 Regression (Savings Amount)", "🌟 Feature Importance"])

    with tab1:
        st.markdown("### 🏷️ Predicting Savings Risk (High / Medium / Low)")
        st.markdown(f"**Best Model: `{meta['best_clf']}`** with **{meta['best_clf_acc']*100:.2f}% accuracy**")
        st.markdown("---")

        # Model comparison
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            clf_df = pd.DataFrame(meta["clf_results"]).T.reset_index()
            clf_df.columns = ["Model", "Test Accuracy", "CV Mean", "CV Std"]
            fig, ax = dark_fig(8, 4)
            colors_bar = [CLR_GREEN if n == meta["best_clf"] else CLR_BLUE for n in clf_df["Model"]]
            bars = ax.barh(clf_df["Model"], clf_df["Test Accuracy"], color=colors_bar, edgecolor="#0f1117")
            for bar, val in zip(bars, clf_df["Test Accuracy"]):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", color="#c9d1d9", fontsize=9)
            ax.set_xlim(0, 1.05); ax.set_xlabel("Accuracy")
            ax.set_title("Classifier Accuracy Comparison")
            star_patch = mpatches.Patch(color=CLR_GREEN, label="Best Model")
            ax.legend(handles=[star_patch], facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            st.markdown("**Model Scores**")
            st.dataframe(clf_df.set_index("Model"), use_container_width=True)

        # Confusion matrix
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            cm     = np.array(meta["cm"])
            labels = meta["classes"]
            fig, ax = dark_fig(6, 5)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=labels, yticklabels=labels,
                        linewidths=0.5, cbar_kws={"shrink":0.8})
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            ax.tick_params(axis="x", rotation=20); ax.tick_params(axis="y", rotation=0)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col4:
            st.markdown("**Classification Report**")
            report = meta["report"]
            rows   = []
            for cls in meta["classes"]:
                r = report.get(cls, {})
                rows.append({"Class": cls, "Precision": round(r.get("precision",0),3),
                             "Recall": round(r.get("recall",0),3), "F1": round(r.get("f1-score",0),3),
                             "Support": int(r.get("support",0))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.metric("Overall Accuracy", f"{report['accuracy']*100:.2f}%")
            st.metric("Macro F1-Score",   f"{report['macro avg']['f1-score']*100:.2f}%")

    with tab2:
        st.markdown("### 📉 Predicting Monthly Savings Amount (₹)")
        best_reg_name = meta["best_reg"]
        best_r2       = meta["best_reg_r2"]
        st.markdown(f"**Best Model: `{best_reg_name}`** with **R² = {best_r2:.4f}**")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            reg_df = pd.DataFrame([
                {"Model": k, "R²": v["R2"], "MAE (₹)": v["MAE"], "RMSE (₹)": v["RMSE"]}
                for k, v in meta["reg_results"].items()
            ])
            st.dataframe(reg_df, use_container_width=True)

            fig, ax = dark_fig(7, 4)
            colors_bar = [CLR_GREEN if n == best_reg_name else CLR_BLUE for n in reg_df["Model"]]
            ax.barh(reg_df["Model"], reg_df["R²"], color=colors_bar, edgecolor="#0f1117")
            ax.set_xlabel("R² Score"); ax.set_title("Regressor R² Comparison")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            fig, ax = dark_fig(7, 5)
            ax.scatter(preds["actual_savings"], preds["predicted_savings"],
                       alpha=0.4, color=CLR_BLUE, s=15)
            lim = [0, max(preds["actual_savings"].max(), preds["predicted_savings"].max())]
            ax.plot(lim, lim, color=CLR_RED, linestyle="--", linewidth=2, label="Perfect fit")
            ax.set_xlabel("Actual Monthly Savings (₹)"); ax.set_ylabel("Predicted (₹)")
            ax.set_title("Actual vs Predicted Savings")
            ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        fig, ax = dark_fig(12, 4)
        residuals = preds["actual_savings"] - preds["predicted_savings"]
        ax.hist(residuals, bins=40, color=CLR_PURPLE, edgecolor="#0f1117", alpha=0.85)
        ax.axvline(0, color=CLR_RED, linestyle="--", linewidth=2)
        ax.set_xlabel("Residual (₹)"); ax.set_ylabel("Count"); ax.set_title("Residual Distribution")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("### 🌟 Top 15 Feature Importances")
        st.markdown(f"*From best classifier: {meta['best_clf']}*")
        fig, ax = dark_fig(10, 7)
        fi_top = feat_imp.head(15).sort_values("importance")
        colors_fi = sns.color_palette("viridis", len(fi_top))
        ax.barh(fi_top["feature"], fi_top["importance"], color=colors_fi, edgecolor="#0f1117")
        ax.set_xlabel("Importance Score"); ax.set_title("Feature Importance — Top 15")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.dataframe(feat_imp.reset_index(drop=True), use_container_width=True)


# =============================================================================
# PAGE 6 — LIVE PREDICTOR
# =============================================================================
elif page == "🔮 Live Predictor":
    st.markdown("# 🔮 Live Finance Predictor")
    st.markdown("Enter a user's financial profile to predict their **Savings Risk** and **Monthly Savings**")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("### 📝 User Profile")
        age_in    = st.slider("Age", 22, 65, 32)
        gender_in = st.selectbox("Gender", ["Male", "Female", "Other"])
        edu_in    = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        occ_in    = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Freelancer", "Business Owner", "Student"])
        city_in   = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        inc_in    = st.number_input("Monthly Income (₹)", 10000, 200000, 55000, step=1000)
        st.markdown("#### 💸 Monthly Expenses (₹)")
        c1, c2 = st.columns(2)
        rent_in   = c1.number_input("Housing Rent",   0, 80000, 12000, step=500)
        groc_in   = c2.number_input("Groceries",      0, 20000,  5000, step=500)
        util_in   = c1.number_input("Utilities",      0, 10000,  2000, step=200)
        trans_in  = c2.number_input("Transport",      0, 15000,  3000, step=200)
        dine_in   = c1.number_input("Dining Out",     0, 20000,  3000, step=500)
        ent_in    = c2.number_input("Entertainment",  0, 15000,  2000, step=200)
        health_in = c1.number_input("Healthcare",     0, 10000,  1500, step=200)
        edu_exp   = c2.number_input("Education Exp",  0, 20000,  1000, step=500)
        shop_in   = c1.number_input("Shopping",       0, 30000,  5000, step=500)
        invest_in = c2.number_input("Investments",    0, 50000,  5000, step=500)
        loan_in   = c1.number_input("Loan EMI",       0, 50000,     0, step=500)
        misc_in   = c2.number_input("Miscellaneous",  0, 10000,  1000, step=200)

        predict_btn = st.button("🔮 Predict Now", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Prediction Results")
        if predict_btn:
            total_exp = rent_in + groc_in + util_in + trans_in + dine_in + ent_in + health_in + edu_exp + shop_in + loan_in + misc_in
            savings   = max(0, inc_in - total_exp - invest_in)
            exp_ratio = round(total_exp / max(inc_in, 1) * 100, 2)
            inv_ratio = round(invest_in / max(inc_in, 1) * 100, 2)
            has_loan  = 1 if loan_in > 0 else 0
            credit_est = int((500 + (savings / max(inc_in,1) * 100) * 3 - exp_ratio * 0.5 + inc_in / 1000 - has_loan * 30))
            credit_est = max(300, min(900, credit_est))

            input_df = pd.DataFrame([{
                "age": age_in, "gender": gender_in, "education": edu_in, "occupation": occ_in,
                "city_tier": city_in, "monthly_income": inc_in, "housing_rent": rent_in,
                "groceries": groc_in, "utilities": util_in, "transport": trans_in,
                "dining_out": dine_in, "entertainment": ent_in, "healthcare": health_in,
                "education_expense": edu_exp, "shopping": shop_in, "investments": invest_in,
                "loan_emi": loan_in, "miscellaneous": misc_in, "total_expenses": total_exp,
                "expense_ratio": exp_ratio, "investment_ratio": inv_ratio,
                "has_loan": has_loan, "credit_score": credit_est,
            }])

            try:
                risk_enc  = clf_model.predict(input_df)[0]
                risk_pred = label_enc.inverse_transform([risk_enc])[0]
                risk_proba = clf_model.predict_proba(input_df)[0]
                sav_pred  = float(reg_model.predict(input_df)[0])

                # Risk badge
                badge_class = {"High Risk": "badge-high", "Medium Risk": "badge-medium", "Low Risk": "badge-low"}[risk_pred]
                st.markdown(f"<br>", unsafe_allow_html=True)
                st.markdown(f"**🎯 Predicted Savings Risk:**  <span class='{badge_class}'>{risk_pred}</span>", unsafe_allow_html=True)
                st.markdown(f"**💰 Predicted Monthly Savings:**  ₹{sav_pred:,.0f}")
                st.markdown("---")

                # KPIs
                kc1, kc2, kc3 = st.columns(3)
                kc1.metric("Total Expenses",  fmt_inr(total_exp))
                kc2.metric("Expense Ratio",   f"{exp_ratio:.1f}%")
                kc3.metric("Estimated Credit",f"{credit_est}")
                kc1.metric("Calculated Savings", fmt_inr(savings))
                kc2.metric("Investment Ratio",   f"{inv_ratio:.1f}%")
                kc3.metric("Has Loan",            "Yes" if has_loan else "No")

                st.markdown("---")
                # Probability bar
                st.markdown("**🎲 Risk Class Probabilities**")
                proba_df = pd.DataFrame({"Class": label_enc.classes_, "Probability": risk_proba})
                fig, ax = dark_fig(7, 3)
                colors_p = [RISK_COLORS.get(c, CLR_BLUE) for c in proba_df["Class"]]
                bars = ax.barh(proba_df["Class"], proba_df["Probability"], color=colors_p, edgecolor="#0f1117")
                for bar, val in zip(bars, proba_df["Probability"]):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.2%}", va="center", color="#c9d1d9", fontsize=10, fontweight="bold")
                ax.set_xlim(0, 1.15); ax.set_xlabel("Probability")
                ax.set_title("Prediction Confidence")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

                # Expense donut
                st.markdown("**💸 Your Expense Breakdown**")
                exp_items = {"Housing": rent_in, "Groceries": groc_in, "Shopping": shop_in,
                             "Dining": dine_in, "Transport": trans_in, "Loan EMI": loan_in,
                             "Entertainment": ent_in, "Healthcare": health_in,
                             "Utilities": util_in, "Education": edu_exp, "Misc": misc_in}
                exp_items = {k: v for k, v in exp_items.items() if v > 0}
                if exp_items:
                    fig, ax = dark_fig(6, 5)
                    wedges, texts, autotexts = ax.pie(
                        list(exp_items.values()), labels=list(exp_items.keys()),
                        autopct="%1.1f%%", startangle=140,
                        colors=sns.color_palette("tab10", len(exp_items)),
                        wedgeprops={"edgecolor":"#0f1117","linewidth":1.5},
                        textprops={"fontsize":8}
                    )
                    for t in texts:     t.set_color("#c9d1d9")
                    for a in autotexts: a.set_color("#ffffff"); a.set_fontsize(7)
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.info("👈 Fill in the profile and click **Predict Now**")
            st.markdown("""
            **What this predicts:**
            - 🏷️ **Savings Risk Category** — High / Medium / Low  
            - 💰 **Monthly Savings Amount** — in ₹
            - 📊 **Confidence probabilities** for each risk class
            - 💸 **Personal expense breakdown** donut chart
            """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#444; font-size:0.78rem'>"
    "💰 Personal Finance Analytics & Prediction System | Built with Python, Scikit-learn & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
