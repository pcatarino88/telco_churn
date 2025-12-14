import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import shap


# -------------------------
# Config
# -------------------------
PIPE_PATH = "models/best_churn_pipeline.pkl"
DATA_PATH = "data/final_dfs/df_final.parquet"
TARGET_COL = "Churn Value"
DEFAULT_THRESHOLD = 0.60

COL_REFERRALS = "Number of Referrals"
COL_CONTRACT = "Contract"
COL_PAPERLESS = "Paperless Billing"

MONTHLY = "Month-to-Month"
ONE_YEAR = "One Year"
TWO_YEAR = "Two Year"

st.set_page_config(page_title="Churn What-if Simulator", layout="wide")
st.title("Churn What-if Simulator")


# -------------------------
# Load assets
# -------------------------
@st.cache_resource
def load_pipe():
    return joblib.load(PIPE_PATH)

@st.cache_data
def load_df():
    return pd.read_parquet(DATA_PATH)

pipe = load_pipe()
df = load_df()

X_base = df.drop(columns=[TARGET_COL], errors="ignore")


# -------------------------
# Helpers
# -------------------------
def predict_proba(pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]

def churn_rate(proba: np.ndarray, threshold: float) -> float:
    return float((proba >= threshold).mean())

def minimal_flip_indices(candidates: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Pick k indices from candidates without replacement (or fewer if not enough)."""
    if k <= 0 or len(candidates) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    k = min(k, len(candidates))
    return rng.choice(candidates, size=k, replace=False)

def apply_referrals_hierarchical(
    series: pd.Series,
    p_has_referrals_target: float,
    p_ge2_given_has_target: float,
    seed: int = 42
) -> pd.Series:
    """
    Adjust referrals with minimal changes:
    - Stage 1: set share of customers with referrals (>=1) by flipping 0 <-> 1
    - Stage 2: among customers with >=1, set share with >=2 by flipping 1 <-> 2
    This keeps distribution close and avoids reshuffling everything.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0).round().astype(int).copy()
    n = len(s)

    # --- Stage 1: target # with >=1 ---
    cur_has = (s >= 1).sum()
    tgt_has = int(round(p_has_referrals_target * n))
    diff = tgt_has - cur_has

    if diff > 0:
        # need more with >=1: flip some 0 -> 1
        idx0 = np.where(s.to_numpy() == 0)[0]
        pick = minimal_flip_indices(idx0, diff, seed=seed)
        s.iloc[pick] = 1
    elif diff < 0:
        # need fewer with >=1: flip some 1 -> 0 (prefer those with exactly 1)
        idx1 = np.where(s.to_numpy() == 1)[0]
        pick = minimal_flip_indices(idx1, -diff, seed=seed)
        s.iloc[pick] = 0
        # If not enough 1s, we could also downgrade 2+ to 0, but keep it simple.

    # --- Stage 2: among >=1, target # with >=2 ---
    has_mask = (s >= 1).to_numpy()
    n_has = int(has_mask.sum())
    if n_has == 0:
        return s

    cur_ge2 = (s[has_mask] >= 2).sum()
    tgt_ge2 = int(round(p_ge2_given_has_target * n_has))
    diff2 = tgt_ge2 - cur_ge2

    if diff2 > 0:
        # need more >=2: flip some 1 -> 2
        idx1 = np.where(s.to_numpy() == 1)[0]
        pick = minimal_flip_indices(idx1, diff2, seed=seed + 1)
        s.iloc[pick] = 2
    elif diff2 < 0:
        # need fewer >=2: flip some 2 -> 1 (prefer exactly 2)
        idx2 = np.where(s.to_numpy() == 2)[0]
        pick = minimal_flip_indices(idx2, -diff2, seed=seed + 1)
        s.iloc[pick] = 1
        # (If you have many 3,4,5... you can extend this; for now keep simple.)

    return s

def apply_contract_hierarchical(
    series: pd.Series,
    p_monthly_target: float,
    p_two_year_given_annual_target: float,
    seed: int = 42
) -> pd.Series:
    """
    Adjust contract with minimal changes:
    - Stage 1: set share of Month-to-Month by moving customers between monthly <-> annual
      (we move from the majority annual type to keep minimal disruption)
    - Stage 2: among annual (One Year + Two Year), set share of Two Year by swapping One Year <-> Two Year
    """
    s = series.astype(str).copy()
    n = len(s)

    # Treat unknown labels as-is, but only operate on known ones.
    known_mask = s.isin([MONTHLY, ONE_YEAR, TWO_YEAR]).to_numpy()
    if known_mask.sum() == 0:
        return s

    # Work on known subset indices
    idx_known = np.where(known_mask)[0]
    s_known = s.iloc[idx_known].copy()
    nk = len(s_known)

    # --- Stage 1: target monthly count ---
    cur_monthly = (s_known == MONTHLY).sum()
    tgt_monthly = int(round(p_monthly_target * nk))
    diff = tgt_monthly - cur_monthly

    if diff > 0:
        # need more monthly: convert some annual -> monthly
        idx_annual = np.where((s_known.to_numpy() == ONE_YEAR) | (s_known.to_numpy() == TWO_YEAR))[0]
        pick_local = minimal_flip_indices(idx_annual, diff, seed=seed)
        s_known.iloc[pick_local] = MONTHLY
    elif diff < 0:
        # need fewer monthly: convert some monthly -> annual
        idx_m = np.where(s_known.to_numpy() == MONTHLY)[0]
        pick_local = minimal_flip_indices(idx_m, -diff, seed=seed)

        # choose where to send them: preserve current annual mix by sending to majority annual class
        cur_one = (s_known == ONE_YEAR).sum()
        cur_two = (s_known == TWO_YEAR).sum()
        send_to = TWO_YEAR if cur_two >= cur_one else ONE_YEAR
        s_known.iloc[pick_local] = send_to

    # --- Stage 2: among annual, target two-year share ---
    annual_mask = (s_known != MONTHLY).to_numpy()
    n_annual = int(annual_mask.sum())
    if n_annual == 0:
        s.iloc[idx_known] = s_known
        return s

    cur_two = (s_known[annual_mask] == TWO_YEAR).sum()
    tgt_two = int(round(p_two_year_given_annual_target * n_annual))
    diff2 = tgt_two - cur_two

    if diff2 > 0:
        # need more two-year: flip some one-year -> two-year
        idx_one = np.where(s_known.to_numpy() == ONE_YEAR)[0]
        pick_local = minimal_flip_indices(idx_one, diff2, seed=seed + 1)
        s_known.iloc[pick_local] = TWO_YEAR
    elif diff2 < 0:
        # need fewer two-year: flip some two-year -> one-year
        idx_two = np.where(s_known.to_numpy() == TWO_YEAR)[0]
        pick_local = minimal_flip_indices(idx_two, -diff2, seed=seed + 1)
        s_known.iloc[pick_local] = ONE_YEAR

    s.iloc[idx_known] = s_known
    return s


# -------------------------
# UI
# -------------------------
tab1, tab2 = st.tabs(["🔄 What-if Simulation", "📊 Model & Insights"])

with st.sidebar:
    st.header("Controls")
    threshold = st.slider(
        "Churn threshold (for churn-rate display)",
        min_value=0.05, max_value=0.95, value=float(DEFAULT_THRESHOLD), step=0.01
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.divider()
    st.subheader("Scenario toggles (use 1 or all)")
    use_referrals = st.checkbox("Simulate Referrals (hierarchical)", value=True)
    use_contract = st.checkbox("Simulate Contract Mix (hierarchical)", value=True)
    use_paperless = st.checkbox("Simulate Paperless Billing", value=True)


with tab1:
    st.subheader("🧪 Scenario builder")

    # Baseline prediction
    base_proba = predict_proba(pipe, X_base)

    # Baseline stats — Referrals
    referrals_base = None
    p_has_ref_base = None
    p_ge2_given_has_base = None
    if COL_REFERRALS in X_base.columns:
        referrals_base = pd.to_numeric(X_base[COL_REFERRALS], errors="coerce").fillna(0).round().astype(int)
        p_has_ref_base = float((referrals_base >= 1).mean())
        has_mask = (referrals_base >= 1)
        p_ge2_given_has_base = float((referrals_base[has_mask] >= 2).mean()) if has_mask.any() else 0.0

    # Baseline stats — Contract
    contract_base = None
    p_monthly_base = None
    p_two_given_annual_base = None
    if COL_CONTRACT in X_base.columns:
        contract_base = X_base[COL_CONTRACT].astype(str)
        known = contract_base.isin([MONTHLY, ONE_YEAR, TWO_YEAR])
        if known.any():
            ck = contract_base[known]
            p_monthly_base = float((ck == MONTHLY).mean())
            annual = ck[ck != MONTHLY]
            p_two_given_annual_base = float((annual == TWO_YEAR).mean()) if len(annual) > 0 else 0.0

    # Baseline stats — Paperless
    paperless_yes_base = None
    if COL_PAPERLESS in X_base.columns:
        paperless_yes_base = float((X_base[COL_PAPERLESS].astype(str) == "Yes").mean())

    # UI controls (defaults = baseline)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Referrals")
        if referrals_base is None:
            st.info(f"Column '{COL_REFERRALS}' not found.")
            p_has_ref_target = 0.0
            p_ge2_given_has_target = 0.0
        else:
            st.write(f"Baseline: **{p_has_ref_base:.1%}** have ≥1 referral")
            st.write(f"Among those: **{p_ge2_given_has_base:.1%}** have ≥2 referrals")

            p_has_ref_target = st.slider(
                "Target % with ≥1 referral",
                0.0, 100.0,
                value=float(round(p_has_ref_base * 100, 1)),
                step=0.5,
                disabled=not use_referrals
            ) / 100.0

            p_ge2_given_has_target = st.slider(
                "Target % with ≥2 (given ≥1)",
                0.0, 100.0,
                value=float(round(p_ge2_given_has_base * 100, 1)),
                step=0.5,
                disabled=not use_referrals
            ) / 100.0

    with c2:
        st.markdown("#### Contract Duration")
        if contract_base is None or p_monthly_base is None:
            st.info(f"Column '{COL_CONTRACT}' not found (or unexpected labels).")
            p_monthly_target = 0.0
            p_two_given_annual_target = 0.0
        else:
            st.write(f"Baseline: **{p_monthly_base:.1%}** Month-to-Month")
            st.write(f"Among annual: **{p_two_given_annual_base:.1%}** Two Year")

            p_monthly_target = st.slider(
                "Target % Month-to-Month",
                0.0, 100.0,
                value=float(round(p_monthly_base * 100, 1)),
                step=0.5,
                disabled=not use_contract
            ) / 100.0

            p_two_given_annual_target = st.slider(
                "Target % Two Year (given annual)",
                0.0, 100.0,
                value=float(round(p_two_given_annual_base * 100, 1)),
                step=0.5,
                disabled=not use_contract
            ) / 100.0

    with c3:
        st.markdown("#### Paperless Billing")
        if paperless_yes_base is None:
            st.info(f"Column '{COL_PAPERLESS}' not found.")
            paperless_yes_target = None
        else:
            st.write(f"Baseline 'Yes': **{paperless_yes_base:.1%}**")
            st.write(f"Baseline 'No': **{1 - paperless_yes_base:.1%}**")
            paperless_yes_target = st.slider(
                "Target % 'Yes'",
                0.0, 100.0,
                value=float(round(paperless_yes_base * 100, 1)),
                step=0.5,
                disabled=not use_paperless
            ) / 100.0

    # Build scenario from baseline with minimal changes
    X_scenario = X_base.copy()

    if use_referrals and referrals_base is not None:
        X_scenario[COL_REFERRALS] = apply_referrals_hierarchical(
            X_scenario[COL_REFERRALS],
            p_has_referrals_target=p_has_ref_target,
            p_ge2_given_has_target=p_ge2_given_has_target,
            seed=int(seed),
        )

    if use_contract and contract_base is not None and p_monthly_base is not None:
        X_scenario[COL_CONTRACT] = apply_contract_hierarchical(
            X_scenario[COL_CONTRACT],
            p_monthly_target=p_monthly_target,
            p_two_year_given_annual_target=p_two_given_annual_target,
            seed=int(seed),
        )

    if use_paperless and paperless_yes_base is not None and paperless_yes_target is not None:
        # Minimal flips for paperless: same helper as before but inlined here
        s = X_scenario[COL_PAPERLESS].astype(str).copy()
        n = len(s)
        cur_yes = int((s == "Yes").sum())
        tgt_yes = int(round(paperless_yes_target * n))
        diff = tgt_yes - cur_yes
        if diff != 0:
            if diff > 0:
                idx = np.where(s.to_numpy() == "No")[0]
                pick = minimal_flip_indices(idx, diff, seed=int(seed) + 10)
                s.iloc[pick] = "Yes"
            else:
                idx = np.where(s.to_numpy() == "Yes")[0]
                pick = minimal_flip_indices(idx, -diff, seed=int(seed) + 10)
                s.iloc[pick] = "No"
        X_scenario[COL_PAPERLESS] = s

    # Predict scenario
    scen_proba = predict_proba(pipe, X_scenario)

    # -------------------------------------------------
    # Summary Results
    # -------------------------------------------------
    st.divider()

    st.subheader("🎯 Impact summary")

    # Metrics
    base_mean = float(np.mean(base_proba))
    scen_mean = float(np.mean(scen_proba))
    base_rate = churn_rate(base_proba, threshold)
    scen_rate = churn_rate(scen_proba, threshold)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Baseline churn rate (≥ {threshold:.2f})", f"{base_rate:.2%}")
    m2.metric(f"Scenario churn rate (≥ {threshold:.2f})", f"{scen_rate:.2%}", f"{(scen_rate-base_rate):+.2%}")
    m3.metric(f"Delta number of churns:", f"{int(round(scen_rate * len(X_base))) - int(round(base_rate * len(X_base))):+,.0f}")

    
    # -------------------------------------------------
    # Visualization of distributions
    # -------------------------------------------------
    st.divider()

    st.subheader("📊 Distribution shift")

    def plot_hist_with_counts(ax, data, bins, title):
        counts, bin_edges, patches = ax.hist(data, bins=bins)
        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Customers")
        ax.set_ylim(0, 3500)

        # Add count labels on top of bars
        for count, patch in zip(counts, patches):
            if count > 0:
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                ax.annotate(
                    f"{int(count)}",
                    (x, y),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 2),
                    textcoords="offset points"
                )

    pA, pB = st.columns(2)

    with pA:
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_hist_with_counts(
            ax,
            base_proba,
            bins=5,
            title="Baseline churn probability"
        )
        st.pyplot(fig)

    with pB:
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_hist_with_counts(
            ax,
            scen_proba,
            bins=5,
            title="Scenario churn probability"
        )
        st.pyplot(fig)

# -------------------------------------------------
# TAB 2
# -------------------------------------------------

with tab2:

    model = pipe.named_steps.get("model", None)
    preprocess = pipe.named_steps.get("preprocess", None)


    # -------------------------------------------------
    # Model Summary
    # -------------------------------------------------

    st.markdown("### ⚙️ Model summary")
    st.write("Model type:", type(model).__name__ if model is not None else "Unknown")
    st.write("Customers:", f"{len(X_base):,}")
    st.write("Key Model Parameters:")
    params = model.get_params()
    
    key_params = {
        "Boosting": params.get("boosting_type"),
        "Trees (n_estimators)": params.get("n_estimators"),
        "Learning rate": params.get("learning_rate"),
        "Max depth": params.get("max_depth"),
        "Num leaves": params.get("num_leaves"),
        "Min child samples": params.get("min_child_samples"),
        "Subsample": params.get("subsample"),
        "Colsample by tree": params.get("colsample_bytree"),
        "Class weight": params.get("class_weight"),
        "Random state": params.get("random_state"),
    }
    
    # Display as a clean table (best for readability)
    params_df = (
        pd.DataFrame.from_dict(key_params, orient="index", columns=["Value"])
        .reset_index()
        .rename(columns={"index": "Parameter"})
    )

    st.dataframe(params_df, use_container_width=True, hide_index=True)


    # -------------------------------------------------
    # Model Results
    # -------------------------------------------------
    st.divider()
    st.markdown("### 📌 Model Results")

    X_eval = X_base.copy()
    y_eval = None
    if TARGET_COL in df.columns:
        y_eval = df[TARGET_COL].astype(int)

    if y_eval is None:
        st.info(f"'{TARGET_COL}' not found in dataset — cannot compute confusion matrix / metrics.")
    else:
        # Probabilities + thresholded predictions
        proba = pipe.predict_proba(X_eval)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        # Confusion matrix + metrics
        cm = confusion_matrix(y_eval, y_pred)
        precision = precision_score(y_eval, y_pred, zero_division=0)
        recall = recall_score(y_eval, y_pred, zero_division=0)
        accuracy = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, zero_division=0)

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown(f"**Confusion Matrix (threshold = {threshold:.2f})**")

            # ⬇️ Smaller figure
            fig, ax = plt.subplots(figsize=(3.8, 3.4))

            color_matrix = np.array([
                ["#2ecc71", "#f5b041"],  # TN, FP
                ["#f5b041", "#2ecc71"]   # FN, TP
            ])

            for i in range(2):
                for j in range(2):
                    ax.add_patch(
                        plt.Rectangle(
                            (j, i), 1, 1,
                            color=color_matrix[i, j],
                            alpha=0.85
                        )
                    )
                    ax.text(
                        j + 0.5, i + 0.5,
                        f"{cm[i, j]:,}",
                        ha="center", va="center",
                        fontsize=12,           # ⬇️ smaller text
                        fontweight="bold",
                        color="black"
                    )

            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)

            ax.set_xticks([0.5, 1.5])
            ax.set_yticks([0.5, 1.5])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=10)
            ax.set_yticklabels(["True 0", "True 1"], fontsize=10)

            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title("Confusion Matrix", fontsize=12)

            ax.tick_params(left=False, bottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.invert_yaxis()

            plt.tight_layout(pad=0.6)  # ⬅️ tighter padding
            st.pyplot(fig, clear_figure=True)

        with right:
            st.markdown("**Key metrics**")
            st.write("")
            st.write("")
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("F1 score", f"{f1:.3f}")

    # -------------------------------------------------
    # Shap Values
    # -------------------------------------------------
    st.divider()

    st.markdown("### 🧠 SHAP summary (feature impact)")

    model = pipe.named_steps.get("model", None)
    preprocess = pipe.named_steps.get("preprocess", None)

    if model is None or preprocess is None:
        st.info("Pipeline does not expose 'model' and/or 'preprocess' steps.")
    else:
        # Choose a sample size (SHAP can be slow on full population)
        max_rows = 2000
        sample_seed = 42

        X_sample = X_base.sample(n=min(max_rows, len(X_base)), random_state=int(sample_seed))

        # Transform raw -> model input
        X_prepped = preprocess.transform(X_sample)

        # Try to get feature names for the plot
        feature_names = None
        if hasattr(preprocess, "important_features"):
            feature_names = list(preprocess.important_features)
        elif hasattr(model, "feature_name_"):
            feature_names = list(model.feature_name_)

        # Make X_prepped a DataFrame if it's numpy
        if not hasattr(X_prepped, "columns"):
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(X_prepped.shape[1])]
            X_prepped = pd.DataFrame(X_prepped, columns=feature_names, index=X_sample.index)

        # SHAP computation
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_prepped)

        # For binary classification, shap_values may be a list [class0, class1]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        shap.summary_plot(sv, X_prepped, show=False, plot_size=None)
        st.pyplot(fig, clear_figure=True)
