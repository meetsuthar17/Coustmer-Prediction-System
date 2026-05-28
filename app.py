import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
import joblib

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CLEAN LIGHT CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* White background everywhere */
.stApp { background-color: #ffffff; }
[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e9ecef; }

/* Remove streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Page title */
.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.95rem;
    color: #6c757d;
    margin-bottom: 1.5rem;
}

/* Section headers */
.section-head {
    font-size: 0.78rem;
    font-weight: 700;
    color: #495057;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    border-bottom: 2px solid #4361ee;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem 0;
    display: inline-block;
}

/* Metric cards */
.kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem; }
.kpi-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    flex: 1; min-width: 120px;
    border-top: 3px solid #4361ee;
}
.kpi-label { font-size: 0.72rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
.kpi-value { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
.kpi-value.good  { color: #2d6a4f; }
.kpi-value.warn  { color: #e07c24; }
.kpi-value.bad   { color: #c1121f; }
.kpi-value.blue  { color: #4361ee; }

/* Comparison table */
.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; border-radius: 8px; overflow: hidden; border: 1px solid #e9ecef; }
.cmp-table th { background: #f1f3f5; color: #495057; padding: 10px 14px; text-align: left; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid #dee2e6; font-weight: 600; }
.cmp-table td { padding: 9px 14px; border-bottom: 1px solid #f1f3f5; color: #212529; }
.cmp-table tr:hover td { background: #f8f9fa; }
.cmp-table tr.best td { background: #f0fff4; }
.cmp-table tr.best td:first-child { border-left: 3px solid #2d6a4f; }
.best-badge { background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 20px; font-size: 0.68rem; font-weight: 600; margin-left: 6px; }
.up   { color: #2d6a4f; font-weight: 600; }
.down { color: #c1121f; font-weight: 600; }
.neu  { color: #6c757d; }
.tag-grid   { background: #dbeafe; color: #1e40af; padding: 2px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 600; }
.tag-random { background: #fef3c7; color: #92400e; padding: 2px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 600; }

/* Predict result */
.result-box { border-radius: 10px; padding: 1.5rem 2rem; text-align: center; margin: 1rem 0; }
.result-churn { background: #fff0f0; border: 2px solid #c1121f; }
.result-safe  { background: #f0fff4; border: 2px solid #2d6a4f; }
.result-title { font-size: 1.4rem; font-weight: 700; margin: 0; }
.result-prob  { font-size: 2.8rem; font-weight: 700; margin: 6px 0; }

/* Buttons */
.stButton > button {
    background: #4361ee;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    transition: background 0.2s;
}
.stButton > button:hover { background: #3451d1; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] { background: transparent; color: #6c757d !important; border-radius: 6px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #ffffff !important; color: #4361ee !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

/* Inputs */
.stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 6px !important;
    color: #212529 !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #f8f9fa !important;
    border: 2px dashed #dee2e6 !important;
    border-radius: 10px !important;
}
.stProgress > div > div { background-color: #4361ee; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df[df["tenure"] > 0].reset_index(drop=True)
    df.drop("customerID", axis=1, inplace=True, errors="ignore")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def encode_df(df):
    """Encode all non-numeric columns using LabelEncoder, regardless of dtype backend."""
    df_enc = df.copy()
    label_encoders = {}
    for col in df_enc.columns:
        try:
            df_enc[col] = df_enc[col].astype(float)  # already numeric — keep as-is
        except (ValueError, TypeError):
            # String column (object, Arrow-backed, or any other) — encode it
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            label_encoders[col] = le
    return df_enc, label_encoders

def score_color(v):
    if v >= 0.90: return "good"
    if v >= 0.75: return "blue"
    if v >= 0.65: return "warn"
    return "bad"

def delta_html(v):
    if v > 0:   return f'<span class="up">▲ +{v:.4f}</span>'
    elif v < 0: return f'<span class="down">▼ {v:.4f}</span>'
    return f'<span class="neu">— {v:.4f}</span>'

def light_plot():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor":   "#ffffff",
        "axes.edgecolor":   "#dee2e6",
        "axes.labelcolor":  "#495057",
        "xtick.color":      "#6c757d",
        "ytick.color":      "#6c757d",
        "text.color":       "#212529",
        "grid.color":       "#f1f3f5",
        "grid.linestyle":   "--",
        "grid.alpha":       0.8,
    })

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Churn Prediction")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Customer-Churn.csv", type=["csv"])
    st.markdown("---")
    st.markdown("**Training Settings**")

    selected_models = st.multiselect(
        "Select models",
        ["Logistic Regression","KNN","Decision Tree","Random Forest",
         "SVM","Naive Bayes","Gradient Boosting","AdaBoost","Extra Trees","XGBoost"],
        default=["Logistic Regression","Random Forest","XGBoost","Gradient Boosting","Extra Trees"]
    )
    run_tuning = st.checkbox("Hyperparameter Tuning", value=True)
    n_iter     = st.slider("RandomSearch iterations", 10, 100, 30, 5) if run_tuning else 30
    cv_folds   = st.slider("CV Folds", 3, 10, 5)
    test_size  = st.slider("Test size (%)", 10, 40, 20) / 100
    st.markdown("---")
    train_btn  = st.button("🚀 Train Models", use_container_width=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<p class="page-title">📊 Customer Churn Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">ML pipeline with Baseline training and Hyperparameter Tuning</p>', unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab_data, tab_eda, tab_baseline, tab_tuned, tab_compare, tab_predict = st.tabs([
    "📂 Data", "📊 EDA", "🤖 Baseline", "🔧 Tuned", "⚔️ Compare", "🎯 Predict"
])

if uploaded_file is None:
    for t in [tab_data, tab_eda, tab_baseline, tab_tuned, tab_compare, tab_predict]:
        with t:
            st.info("⬅️ Upload your Customer-Churn.csv file in the sidebar to begin.")
    st.stop()

df = load_data(uploaded_file)

# ── TAB: DATA ────────────────────────────────────────────────
with tab_data:
    churn_rate = df["Churn"].mean() * 100
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-label">Customers</div><div class="kpi-value blue">{len(df):,}</div></div>
        <div class="kpi-card"><div class="kpi-label">Features</div><div class="kpi-value blue">{df.shape[1]-1}</div></div>
        <div class="kpi-card"><div class="kpi-label">Churn Rate</div><div class="kpi-value warn">{churn_rate:.1f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">Missing Values</div><div class="kpi-value good">0</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="section-head">Data Preview</span>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=380)
    st.markdown('<span class="section-head">Descriptive Statistics</span>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

# ── TAB: EDA ─────────────────────────────────────────────────
with tab_eda:
    light_plot()
    BLUE, RED, GRAY = "#4361ee", "#ef233c", "#adb5bd"

    st.markdown('<span class="section-head">Class Distribution</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        counts = df["Churn"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(["No Churn","Churn"], counts.values, color=[BLUE, RED], width=0.45, edgecolor="white")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+30, str(v), ha="center", fontsize=11, fontweight="600")
        ax.set_title("Churn Count", fontsize=12, fontweight="600", pad=10)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.pie(counts.values, labels=["No Churn","Churn"], autopct="%1.1f%%",
               colors=[BLUE, RED], startangle=140,
               wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title("Churn Split", fontsize=12, fontweight="600", pad=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<span class="section-head">Categorical Features vs Churn</span>', unsafe_allow_html=True)
    cat_cols = [c for c in ["gender","Contract","PaymentMethod","InternetService","TechSupport"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        no_ch = df[df["Churn"]==0][col].value_counts()
        ch    = df[df["Churn"]==1][col].value_counts()
        labels = list(no_ch.index.union(ch.index))
        x = np.arange(len(labels)); w = 0.38
        axes[i].bar(x-w/2, [no_ch.get(l,0) for l in labels], w, label="No Churn", color=BLUE, alpha=0.85)
        axes[i].bar(x+w/2, [ch.get(l,0)    for l in labels], w, label="Churn",    color=RED,  alpha=0.85)
        axes[i].set_xticks(x); axes[i].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axes[i].set_title(col, fontsize=10, fontweight="600")
        axes[i].legend(fontsize=8); axes[i].spines[["top","right"]].set_visible(False)
    axes[-1].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<span class="section-head">Numerical Features</span>', unsafe_allow_html=True)
    num_cols = [c for c in ["tenure","MonthlyCharges","TotalCharges"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for i, col in enumerate(num_cols):
        axes[0][i].hist(df[df["Churn"]==0][col], bins=30, color=BLUE, alpha=0.7, label="No Churn")
        axes[0][i].hist(df[df["Churn"]==1][col], bins=30, color=RED,  alpha=0.7, label="Churn")
        axes[0][i].set_title(f"{col} Distribution", fontsize=10, fontweight="600")
        axes[0][i].legend(fontsize=8); axes[0][i].spines[["top","right"]].set_visible(False)
        bp = axes[1][i].boxplot(
            [df[df["Churn"]==0][col].dropna(), df[df["Churn"]==1][col].dropna()],
            patch_artist=True, labels=["No Churn","Churn"]
        )
        for patch, color in zip(bp["boxes"], [BLUE, RED]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        axes[1][i].set_title(f"{col} by Churn", fontsize=10, fontweight="600")
        axes[1][i].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<span class="section-head">Correlation Heatmap</span>', unsafe_allow_html=True)
    df_enc, _ = encode_df(df)
    corr = df_enc.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", mask=mask,
                ax=ax, linewidths=0.4, annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="600", pad=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── TRAINING ─────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False

if train_btn:
    if not selected_models:
        st.error("Select at least one model."); st.stop()

    df_enc, label_encoders = encode_df(df)
    X = df_enc.drop("Churn", axis=1)
    y = df_enc["Churn"]
    # For every column: try casting to float directly.
    # If it fails (string values present), encode with LabelEncoder regardless of dtype.
    for col in X.columns:
        try:
            X[col] = X[col].astype(float)
        except (ValueError, TypeError):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)

    ALL_MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(probability=True, class_weight="balanced", random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "AdaBoost":            AdaBoostClassifier(random_state=42),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
    }
    PARAM_GRIDS = {
        "Logistic Regression": {"method":"grid","model":LogisticRegression(max_iter=1000,random_state=42),
            "params":{"C":[0.01,0.1,1,10,100],"solver":["lbfgs","liblinear"],"penalty":["l2"]}},
        "Random Forest":       {"method":"random","model":RandomForestClassifier(random_state=42),
            "params":{"n_estimators":randint(100,500),"max_depth":[None,5,10,20,30],
                      "min_samples_split":randint(2,20),"min_samples_leaf":randint(1,10),"max_features":["sqrt","log2",None]}},
        "Gradient Boosting":   {"method":"random","model":GradientBoostingClassifier(random_state=42),
            "params":{"n_estimators":randint(100,400),"learning_rate":uniform(0.01,0.3),
                      "max_depth":[3,4,5,6,7],"subsample":uniform(0.6,0.4),"min_samples_split":randint(2,20)}},
        "XGBoost":             {"method":"random","model":XGBClassifier(eval_metric="logloss",random_state=42,verbosity=0),
            "params":{"n_estimators":randint(100,400),"learning_rate":uniform(0.01,0.3),"max_depth":randint(3,10),
                      "subsample":uniform(0.6,0.4),"colsample_bytree":uniform(0.6,0.4),"gamma":uniform(0,0.5),
                      "reg_alpha":uniform(0,1),"reg_lambda":uniform(0.5,2)}},
        "Extra Trees":         {"method":"random","model":ExtraTreesClassifier(random_state=42),
            "params":{"n_estimators":randint(100,400),"max_depth":[None,10,20,30],
                      "min_samples_split":randint(2,20),"min_samples_leaf":randint(1,10),"max_features":["sqrt","log2",None]}},
        "Decision Tree":       {"method":"grid","model":DecisionTreeClassifier(random_state=42),
            "params":{"max_depth":[3,5,7,10,None],"min_samples_split":[2,5,10,20],"min_samples_leaf":[1,2,4,8],"criterion":["gini","entropy"]}},
        "KNN":                 {"method":"grid","model":KNeighborsClassifier(),
            "params":{"n_neighbors":[3,5,7,9,11,15],"weights":["uniform","distance"],"metric":["euclidean","manhattan"]}},
        "AdaBoost":            {"method":"random","model":AdaBoostClassifier(random_state=42),
            "params":{"n_estimators":randint(50,300),"learning_rate":uniform(0.01,1.5)}},
    }
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    total = len(selected_models)
    prog  = st.progress(0, text="Starting training...")

    # Baseline
    baseline_results, baseline_models_trained = [], {}
    for idx, name in enumerate(selected_models):
        prog.progress(idx / (total * (2 if run_tuning else 1)), text=f"Baseline: {name}…")
        model = ALL_MODELS[name]
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        cv_f1  = cross_val_score(model, X_train_sm, y_train_sm, cv=cv_strategy, scoring="f1")
        baseline_results.append({
            "Model":      name,
            "Accuracy":   round(accuracy_score(y_test, y_pred), 4),
            "Precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score":   round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC":    round(roc_auc_score(y_test, y_prob), 4),
            "PR-AUC":     round(average_precision_score(y_test, y_prob), 4),
            "CV F1 Mean": round(cv_f1.mean(), 4),
        })
        baseline_models_trained[name] = model

    # Tuning
    tuned_results, tuned_models_trained = [], {}
    if run_tuning:
        for idx, name in enumerate(selected_models):
            prog.progress((total + idx) / (total * 2), text=f"Tuning: {name}…")
            if name not in PARAM_GRIDS:
                tuned_results.append({**baseline_results[idx], "CV ROC-AUC": baseline_results[idx]["ROC-AUC"], "Best Params":"N/A","Method":"—"})
                tuned_models_trained[name] = baseline_models_trained[name]; continue
            cfg = PARAM_GRIDS[name]
            if cfg["method"] == "grid":
                searcher = GridSearchCV(cfg["model"], cfg["params"], cv=cv_strategy, scoring="roc_auc", n_jobs=-1, verbose=0)
            else:
                searcher = RandomizedSearchCV(cfg["model"], cfg["params"], n_iter=n_iter, cv=cv_strategy, scoring="roc_auc", random_state=42, n_jobs=-1, verbose=0)
            searcher.fit(X_train_sm, y_train_sm)
            best_mdl = searcher.best_estimator_
            y_pred   = best_mdl.predict(X_test_sc)
            y_prob   = best_mdl.predict_proba(X_test_sc)[:, 1]
            tuned_results.append({
                "Model":      name,
                "Accuracy":   round(accuracy_score(y_test, y_pred), 4),
                "Precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1 Score":   round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC":    round(roc_auc_score(y_test, y_prob), 4),
                "PR-AUC":     round(average_precision_score(y_test, y_prob), 4),
                "CV ROC-AUC": round(searcher.best_score_, 4),
                "Best Params":str(searcher.best_params_),
                "Method":     cfg["method"].upper() + "SearchCV",
            })
            tuned_models_trained[name] = best_mdl

    prog.progress(1.0, text="✅ Done!"); time.sleep(0.5); prog.empty()
    st.session_state.update({
        "trained": True, "baseline_results": baseline_results, "tuned_results": tuned_results,
        "baseline_models": baseline_models_trained, "tuned_models": tuned_models_trained,
        "X_test_sc": X_test_sc, "y_test": y_test, "X_columns": list(X.columns),
        "scaler": scaler, "run_tuning": run_tuning, "df_enc": df_enc,
        "label_encoders": label_encoders,
    })
    st.success("✅ Training complete! Explore the tabs above.")

# ── RESULTS ──────────────────────────────────────────────────
if st.session_state.get("trained"):
    br  = st.session_state.baseline_results
    tr  = st.session_state.tuned_results
    bm  = st.session_state.baseline_models
    tm  = st.session_state.tuned_models
    Xt  = st.session_state.X_test_sc
    yt  = st.session_state.y_test
    rt  = st.session_state.run_tuning
    BLUE, RED, GREEN = "#4361ee", "#ef233c", "#2d6a4f"

    base_df  = pd.DataFrame(br).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    tuned_df = pd.DataFrame(tr).sort_values("ROC-AUC", ascending=False).reset_index(drop=True) if rt else None

    def results_table(df_res, extra_cols=None):
        metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","PR-AUC"] + (extra_cols or [])
        best_roc = df_res["ROC-AUC"].max()
        rows = ""
        for _, row in df_res.iterrows():
            is_best = row["ROC-AUC"] == best_roc
            cls     = "best" if is_best else ""
            badge   = '<span class="best-badge">BEST</span>' if is_best else ""
            cells   = f'<td>{row["Model"]}{badge}</td>'
            for m in metrics:
                v = row[m]
                if isinstance(v, float):
                    cells += f"<td>{v:.4f}</td>"
                else:
                    tag = f'<span class="tag-grid">{v}</span>' if "GRID" in str(v) else f'<span class="tag-random">{v}</span>' if "RANDOM" in str(v) else v
                    cells += f"<td>{tag}</td>"
            rows += f'<tr class="{cls}">{cells}</tr>'
        headers = "".join(f"<th>{m}</th>" for m in ["Model"] + metrics)
        return f'<table class="cmp-table"><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>'

    # ── BASELINE TAB ─────────────────────────────────────────
    with tab_baseline:
        light_plot()
        best = base_df.iloc[0]
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-label">Best Model</div><div class="kpi-value blue" style="font-size:1rem">{best['Model']}</div></div>
            <div class="kpi-card"><div class="kpi-label">ROC-AUC</div><div class="kpi-value {score_color(best['ROC-AUC'])}">{best['ROC-AUC']}</div></div>
            <div class="kpi-card"><div class="kpi-label">F1 Score</div><div class="kpi-value {score_color(best['F1 Score'])}">{best['F1 Score']}</div></div>
            <div class="kpi-card"><div class="kpi-label">Recall</div><div class="kpi-value {score_color(best['Recall'])}">{best['Recall']}</div></div>
            <div class="kpi-card"><div class="kpi-label">Precision</div><div class="kpi-value {score_color(best['Precision'])}">{best['Precision']}</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="section-head">Results Table</span>', unsafe_allow_html=True)
        st.markdown(results_table(base_df, ["CV F1 Mean"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<span class="section-head">Performance Charts</span>', unsafe_allow_html=True)
        plot_metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","PR-AUC"]
        fig, axes = plt.subplots(2, 3, figsize=(17, 8))
        axes = axes.flatten()
        for i, metric in enumerate(plot_metrics):
            sd = base_df.sort_values(metric, ascending=True)
            best_v = sd[metric].max()
            colors = [GREEN if v == best_v else BLUE for v in sd[metric]]
            axes[i].barh(sd["Model"], sd[metric], color=colors, height=0.55)
            axes[i].set_xlim(0, 1.1)
            axes[i].set_title(metric, fontsize=10, fontweight="600")
            axes[i].axvline(best_v, color=RED, linestyle="--", alpha=0.5, linewidth=1)
            axes[i].spines[["top","right"]].set_visible(False)
            for j, val in enumerate(sd[metric]):
                axes[i].text(val+0.005, j, f"{val:.3f}", va="center", fontsize=7.5)
        plt.suptitle("Baseline — All Metrics Comparison", fontsize=13, fontweight="700", y=1.01)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown('<span class="section-head">Confusion Matrices</span>', unsafe_allow_html=True)
        cols = st.columns(min(3, len(bm)))
        for i, (name, model) in enumerate(bm.items()):
            with cols[i % 3]:
                cm = confusion_matrix(yt, model.predict(Xt))
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
                ax.set_title(name, fontsize=9, fontweight="600")
                plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TUNED TAB ────────────────────────────────────────────
    with tab_tuned:
        if not rt:
            st.info("Enable Hyperparameter Tuning in the sidebar and retrain.")
        else:
            light_plot()
            best_t = tuned_df.iloc[0]
            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-label">Best Tuned Model</div><div class="kpi-value blue" style="font-size:1rem">{best_t['Model']}</div></div>
                <div class="kpi-card"><div class="kpi-label">ROC-AUC</div><div class="kpi-value {score_color(best_t['ROC-AUC'])}">{best_t['ROC-AUC']}</div></div>
                <div class="kpi-card"><div class="kpi-label">F1 Score</div><div class="kpi-value {score_color(best_t['F1 Score'])}">{best_t['F1 Score']}</div></div>
                <div class="kpi-card"><div class="kpi-label">Recall</div><div class="kpi-value {score_color(best_t['Recall'])}">{best_t['Recall']}</div></div>
                <div class="kpi-card"><div class="kpi-label">CV ROC-AUC</div><div class="kpi-value {score_color(best_t['CV ROC-AUC'])}">{best_t['CV ROC-AUC']}</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<span class="section-head">Tuned Results Table</span>', unsafe_allow_html=True)
            st.markdown(results_table(tuned_df, ["CV ROC-AUC","Method"]), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<span class="section-head">Best Hyperparameters</span>', unsafe_allow_html=True)
            for _, row in tuned_df.iterrows():
                with st.expander(f"🔧 {row['Model']}"):
                    st.code(row.get("Best Params","N/A"), language="python")

            st.markdown('<span class="section-head">Tuned Confusion Matrices</span>', unsafe_allow_html=True)
            cols = st.columns(min(3, len(tm)))
            for i, (name, model) in enumerate(tm.items()):
                with cols[i % 3]:
                    cm = confusion_matrix(yt, model.predict(Xt))
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"]).plot(ax=ax, cmap="Greens", colorbar=False)
                    ax.set_title(name, fontsize=9, fontweight="600")
                    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── COMPARE TAB ──────────────────────────────────────────
    with tab_compare:
        light_plot()
        st.markdown('<span class="section-head">Baseline vs Tuned — Full Comparison</span>', unsafe_allow_html=True)

        if rt and tuned_df is not None:
            bc = base_df[["Model","ROC-AUC","F1 Score","Recall","Precision","Accuracy"]].rename(
                columns={"ROC-AUC":"Base ROC","F1 Score":"Base F1","Recall":"Base Recall","Precision":"Base Prec","Accuracy":"Base Acc"})
            tc = tuned_df[["Model","ROC-AUC","F1 Score","Recall","Precision","Accuracy"]].rename(
                columns={"ROC-AUC":"Tuned ROC","F1 Score":"Tuned F1","Recall":"Tuned Recall","Precision":"Tuned Prec","Accuracy":"Tuned Acc"})
            cmp = bc.merge(tc, on="Model", how="inner")
            cmp["Δ ROC-AUC"] = (cmp["Tuned ROC"] - cmp["Base ROC"]).round(4)
            cmp["Δ F1"]      = (cmp["Tuned F1"]  - cmp["Base F1"]).round(4)
            cmp["Δ Recall"]  = (cmp["Tuned Recall"] - cmp["Base Recall"]).round(4)
            cmp = cmp.sort_values("Tuned ROC", ascending=False).reset_index(drop=True)

            best_roc = cmp["Tuned ROC"].max()
            rows = ""
            for _, row in cmp.iterrows():
                is_best = row["Tuned ROC"] == best_roc
                cls   = "best" if is_best else ""
                badge = '<span class="best-badge">BEST</span>' if is_best else ""
                rows += f"""<tr class="{cls}">
                    <td>{row['Model']}{badge}</td>
                    <td>{row['Base ROC']:.4f}</td><td>{row['Tuned ROC']:.4f}</td><td>{delta_html(row['Δ ROC-AUC'])}</td>
                    <td>{row['Base F1']:.4f}</td><td>{row['Tuned F1']:.4f}</td><td>{delta_html(row['Δ F1'])}</td>
                    <td>{row['Base Recall']:.4f}</td><td>{row['Tuned Recall']:.4f}</td><td>{delta_html(row['Δ Recall'])}</td>
                </tr>"""
            st.markdown(f"""
            <table class="cmp-table"><thead><tr>
                <th>Model</th>
                <th>Base ROC-AUC</th><th>Tuned ROC-AUC</th><th>Δ</th>
                <th>Base F1</th><th>Tuned F1</th><th>Δ</th>
                <th>Base Recall</th><th>Tuned Recall</th><th>Δ</th>
            </tr></thead><tbody>{rows}</tbody></table>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Before vs After bar
            st.markdown('<span class="section-head">ROC-AUC & F1: Before vs After</span>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            x, w = np.arange(len(cmp)), 0.35
            for ax, bc_col, tc_col, title in [
                (axes[0],"Base ROC","Tuned ROC","ROC-AUC"),
                (axes[1],"Base F1","Tuned F1","F1 Score"),
            ]:
                ax.bar(x-w/2, cmp[bc_col], w, label="Baseline", color=BLUE, alpha=0.8)
                ax.bar(x+w/2, cmp[tc_col], w, label="Tuned",    color=GREEN, alpha=0.8)
                ax.set_xticks(x); ax.set_xticklabels(cmp["Model"], rotation=30, ha="right", fontsize=9)
                ax.set_ylim(0.5, 1.05); ax.set_title(title, fontsize=11, fontweight="600")
                ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
                ax.axhline(0.9, color=RED, linestyle="--", alpha=0.35, linewidth=1)
            plt.suptitle("Hyperparameter Tuning — Performance Improvement", fontsize=13, fontweight="700")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Delta chart
            st.markdown('<span class="section-head">Improvement (Δ) from Tuning</span>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 4))
            x, w = np.arange(len(cmp)), 0.35
            ax.bar(x-w/2, cmp["Δ ROC-AUC"], w, color=[GREEN if v>=0 else RED for v in cmp["Δ ROC-AUC"]], label="Δ ROC-AUC")
            ax.bar(x+w/2, cmp["Δ F1"],      w, color=[BLUE  if v>=0 else "#e07c24" for v in cmp["Δ F1"]],  label="Δ F1 Score", alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(cmp["Model"], rotation=30, ha="right", fontsize=9)
            ax.axhline(0, color="#adb5bd", linewidth=1)
            ax.set_title("Improvement per Model (Tuned − Baseline)", fontsize=11, fontweight="600")
            ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        else:
            st.markdown(results_table(base_df, ["CV F1 Mean"]), unsafe_allow_html=True)

        # ROC Curves
        st.markdown('<span class="section-head">ROC Curves</span>', unsafe_allow_html=True)
        models_plot = tm if (rt and tm) else bm
        palette = ["#4361ee","#2d6a4f","#ef233c","#e07c24","#7209b7","#0077b6","#f77f00","#c77dff","#06d6a0","#ffd60a"]
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, (name, model) in enumerate(models_plot.items()):
            y_prob = model.predict_proba(Xt)[:, 1]
            fpr, tpr, _ = roc_curve(yt, y_prob)
            ax.plot(fpr, tpr, lw=2, color=palette[i % len(palette)], label=f"{name} (AUC={roc_auc_score(yt, y_prob):.3f})")
        ax.plot([0,1],[0,1],"--", color="#adb5bd", linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontsize=12, fontweight="600")
        ax.legend(fontsize=8, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Feature importance
        best_name = (tuned_df.iloc[0]["Model"] if rt and tuned_df is not None else base_df.iloc[0]["Model"])
        best_obj  = (tm.get(best_name) or bm.get(best_name))
        X_cols    = st.session_state.X_columns
        if hasattr(best_obj, "feature_importances_"):
            st.markdown(f'<span class="section-head">Feature Importances — {best_name}</span>', unsafe_allow_html=True)
            fi = pd.Series(best_obj.feature_importances_, index=X_cols).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(fi.index[::-1], fi.values[::-1], color=BLUE, height=0.6)
            ax.set_title(f"Top 15 Features — {best_name}", fontsize=11, fontweight="600")
            ax.set_xlabel("Importance Score"); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── PREDICT TAB ──────────────────────────────────────────
    with tab_predict:
        st.markdown('<span class="section-head">Single Customer Prediction</span>', unsafe_allow_html=True)
        model_choice = st.selectbox("Model", list(tm.keys()) if (rt and tm) else list(bm.keys()))
        pred_model   = (tm.get(model_choice) or bm.get(model_choice))

        st.markdown("**Enter customer details:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender    = st.selectbox("Gender", ["Male","Female"])
            senior    = st.selectbox("Senior Citizen", [0,1])
            partner   = st.selectbox("Partner", ["Yes","No"])
            depend    = st.selectbox("Dependents", ["Yes","No"])
            tenure    = st.slider("Tenure (months)", 0, 72, 12)
            phone_svc = st.selectbox("Phone Service", ["Yes","No"])
        with c2:
            multi_ln  = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
            inet_svc  = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            online_sec= st.selectbox("Online Security", ["Yes","No","No internet service"])
            online_bk = st.selectbox("Online Backup", ["Yes","No","No internet service"])
            dev_prot  = st.selectbox("Device Protection", ["Yes","No","No internet service"])
            tech_sup  = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        with c3:
            stream_tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
            stream_mv = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
            contract  = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes","No"])
            payment   = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly_c = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total_c   = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_c), 1.0)

        if st.button("🔮 Predict Churn", use_container_width=True):
            raw = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": depend, "tenure": tenure, "PhoneService": phone_svc,
                "MultipleLines": multi_ln, "InternetService": inet_svc,
                "OnlineSecurity": online_sec, "OnlineBackup": online_bk,
                "DeviceProtection": dev_prot, "TechSupport": tech_sup,
                "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly_c,
                "TotalCharges": total_c
            }
            stored_encoders = st.session_state.label_encoders
            encoded = {}
            for col, val in raw.items():
                if col in stored_encoders:
                    le = stored_encoders[col]
                    val_str = str(val)
                    encoded[col] = int(le.transform([val_str])[0]) if val_str in le.classes_ else 0
                else:
                    encoded[col] = val
            X_cols = st.session_state.X_columns
            inp_row = {c: [encoded.get(c, 0)] for c in X_cols}
            inp = pd.DataFrame(inp_row).astype(float)
            X_inp = st.session_state.scaler.transform(inp)
            prob  = pred_model.predict_proba(X_inp)[0][1]
            pred  = pred_model.predict(X_inp)[0]

            if pred == 1:
                st.markdown(f"""
                <div class="result-box result-churn">
                    <p class="result-title" style="color:#c1121f">⚠️ HIGH CHURN RISK</p>
                    <p class="result-prob"  style="color:#c1121f">{prob*100:.1f}%</p>
                    <p style="color:#6c757d;margin:0">probability of churning</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-safe">
                    <p class="result-title" style="color:#2d6a4f">✅ LOW CHURN RISK</p>
                    <p class="result-prob"  style="color:#2d6a4f">{prob*100:.1f}%</p>
                    <p style="color:#6c757d;margin:0">probability of churning</p>
                </div>""", unsafe_allow_html=True)

            light_plot()
            fig, ax = plt.subplots(figsize=(8, 1.4))
            ax.barh([""], [prob],     color="#ef233c" if prob > 0.5 else "#2d6a4f", height=0.4)
            ax.barh([""], [1-prob], left=[prob], color="#e9ecef", height=0.4)
            ax.set_xlim(0,1); ax.axvline(0.5, color="#6c757d", linestyle="--", linewidth=1)
            ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(["0%","25%","50%","75%","100%"])
            ax.set_title(f"Churn Probability — {model_choice}", fontsize=10, fontweight="600")
            ax.spines[["top","right","left"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

else:
    for t in [tab_baseline, tab_tuned, tab_compare, tab_predict]:
        with t:
            st.info("Click **🚀 Train Models** in the sidebar to start.")