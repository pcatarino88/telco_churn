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
COL_PAYMENT = "Payment Method"   # ✅ NEW (instead of Paperless)

MONTHLY = "Month-to-Month"
ONE_YEAR = "One Year"
TWO_YEAR = "Two Year"

PAY_BANK = "Bank Withdrawal"
PAY_CC = "Credit Card"
PAY_MAIL = "Mailed Check"

st.set_page_config(page_title="Telco Churn Simulator",  page_icon="⚡", layout="wide")
st.title("⚡Telco Churn Simulator")


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


# Referrals simulation (0 ↔ 2 only)
def apply_referrals_ge2_only(
    series: pd.Series,
    p_ge2_target: float,
    seed: int = 42
) -> pd.Series:
    """
    Minimal-change referrals simulation:
    - Target share of customers with referrals >= 2
    - Only flips between 0 and 2 (no reshuffling)
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0).round().astype(int).copy()
    n = len(s)

    cur_ge2 = int((s >= 2).sum())
    tgt_ge2 = int(round(p_ge2_target * n))
    diff = tgt_ge2 - cur_ge2

    if diff > 0:
        # Need more >=2: flip some 0 -> 2
        idx0 = np.where(s.to_numpy() == 0)[0]
        pick = minimal_flip_indices(idx0, diff, seed=seed)
        s.iloc[pick] = 2

    elif diff < 0:
        # Need fewer >=2: flip some 2 -> 0 (prefer exactly 2)
        idx2 = np.where(s.to_numpy() == 2)[0]
        pick = minimal_flip_indices(idx2, -diff, seed=seed)
        s.iloc[pick] = 0

    return s

# Contract simulation (Month-to-Month ↔ Two Year only)
def apply_contract_two_year_only(
    series: pd.Series,
    p_two_year_target: float,
    seed: int = 42
) -> pd.Series:
    """
    Minimal-change contract simulation:
    - Target share of customers with Two Year contracts
    - Only flips between Month-to-Month and Two Year (no reshuffling)
    """
    s = series.astype(str).copy()
    n = len(s)

    cur_two = int((s == TWO_YEAR).sum())
    tgt_two = int(round(p_two_year_target * n))
    diff = tgt_two - cur_two

    if diff > 0:
        # Need more Two Year: flip some Month-to-Month -> Two Year
        idx_m = np.where(s.to_numpy() == MONTHLY)[0]
        pick = minimal_flip_indices(idx_m, diff, seed=seed)
        s.iloc[pick] = TWO_YEAR

    elif diff < 0:
        # Need fewer Two Year: flip some Two Year -> Month-to-Month
        idx_two = np.where(s.to_numpy() == TWO_YEAR)[0]
        pick = minimal_flip_indices(idx_two, -diff, seed=seed)
        s.iloc[pick] = MONTHLY

    return s

# Payment Method simulation (Credit Card target)
def apply_payment_creditcard_target(
    series: pd.Series,
    target_cc_rate: float,
    seed: int = 42
) -> pd.Series:
    """
    Minimal-change simulation for Payment Method with a target Credit Card proportion.

    Rules:
    - Increasing CC:
        1) convert Bank Withdrawal -> Credit Card
        2) if Bank Withdrawal is 0 (or not enough), convert Mailed Check -> Credit Card
    - Decreasing CC:
        1) convert Credit Card -> Bank Withdrawal (restore BW first)
        2) if needed, convert Credit Card -> Mailed Check
    No reshuffle: only flips a random subset.
    """
    s = series.astype(str).copy()
    n = len(s)

    cur_cc = int((s == PAY_CC).sum())
    tgt_cc = int(round(target_cc_rate * n))
    diff = tgt_cc - cur_cc

    if diff == 0:
        return s

    # Need to INCREASE Credit Card
    if diff > 0:
        # 1) BW -> CC
        idx_bw = np.where(s.to_numpy() == PAY_BANK)[0]
        pick_bw = minimal_flip_indices(idx_bw, diff, seed=seed)
        s.iloc[pick_bw] = PAY_CC

        remaining = diff - len(pick_bw)
        if remaining > 0:
            # 2) Mail -> CC (only if BW exhausted)
            idx_mail = np.where(s.to_numpy() == PAY_MAIL)[0]
            pick_mail = minimal_flip_indices(idx_mail, remaining, seed=seed + 1)
            s.iloc[pick_mail] = PAY_CC

        return s

    # Need to DECREASE Credit Card
    if diff < 0:
        need = -diff

        # 1) CC -> BW (first)
        idx_cc = np.where(s.to_numpy() == PAY_CC)[0]
        pick_to_bw = minimal_flip_indices(idx_cc, need, seed=seed)
        s.iloc[pick_to_bw] = PAY_BANK

        remaining = need - len(pick_to_bw)
        if remaining > 0:
            # 2) CC -> Mail (only if not enough CC found, very rare)
            idx_cc2 = np.where(s.to_numpy() == PAY_CC)[0]
            pick_to_mail = minimal_flip_indices(idx_cc2, remaining, seed=seed + 1)
            s.iloc[pick_to_mail] = PAY_MAIL

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
    use_payment = st.checkbox("Simulate Payment Method", value=True) 


with tab1:
    st.subheader("🧪 Scenario builder")

    # Baseline prediction
    base_proba = predict_proba(pipe, X_base)

    # Baseline stats — Referrals
    referrals_base = None
    p_ge2_base = None
    if COL_REFERRALS in X_base.columns:
        referrals_base = pd.to_numeric(X_base[COL_REFERRALS], errors="coerce").fillna(0).round().astype(int)
        p_ge2_base = float((referrals_base >= 2).mean())

    # Baseline stats — Contract
    contract_base = None
    p_two_year_base = None
    if COL_CONTRACT in X_base.columns:
        contract_base = X_base[COL_CONTRACT].astype(str)
        p_two_year_base = float((contract_base == TWO_YEAR).mean())

    # Baseline stats — Payment Method 
    pay_cc_base = None
    pay_bw_base = None
    pay_mail_base = None
    if COL_PAYMENT in X_base.columns:
        pm = X_base[COL_PAYMENT].astype(str)
        pay_cc_base = float((pm == PAY_CC).mean())
        pay_bw_base = float((pm == PAY_BANK).mean())
        pay_mail_base = float((pm == PAY_MAIL).mean())


    # UI controls (defaults = baseline)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Referrals")
        if referrals_base is None:
            st.info(f"Column '{COL_REFERRALS}' not found.")
            p_ge2_target = 0.25
        else:
            st.write(f"Baseline: **{p_ge2_base:.1%}** customers with ≥2 referrals")

            # Referrals slider
            p_ge2_target = (
                st.slider( 
                    "Simulated proportion of customers with ≥2 referrals",
                    min_value=20.0, max_value=60.0,
                    value=float(round(p_ge2_base * 100, 2)),
                    step=1.0,
                    disabled=not use_referrals
                )
                / 100.0
            )

    with c2:
        st.markdown("#### Contract Duration")
        if contract_base is None:
            st.info(f"Column '{COL_CONTRACT}' not found.")
            p_two_year_target = 0.25
        else:
            st.write(f"Baseline: **{p_two_year_base:.1%}** customers with Two Year contracts")

            # Contract Duration slider
            p_two_year_target = st.slider(
                "Simulated proportion of customers with Two Year contracts",
                min_value=20.0, max_value=60.0,
                value=float(round(p_two_year_base * 100, 2)),
                step=1.0,
                disabled=not use_contract
            ) / 100.0

    with c3:
        st.markdown("#### Payment Method")
        if pay_cc_base is None:
            st.info(f"Column '{COL_PAYMENT}' not found.")
            pay_cc_target = None
        else: 
            st.write(f"Baseline '{PAY_CC}': **{pay_cc_base:.1%}** customers with payment via Credit Card")

            # Payment Method slider
            pay_cc_target = st.slider(
                "Simulated proportion of customers with payment via Credit Card",
                min_value=20.0, max_value=60.0,
                value=float(round(pay_cc_base * 100, 2)),
                step=1.0,
                disabled=not use_payment
            ) / 100.0

    # Build scenario from baseline with minimal changes
    X_scenario = X_base.copy()

    if use_referrals and referrals_base is not None:
        X_scenario[COL_REFERRALS] = apply_referrals_ge2_only(
            X_scenario[COL_REFERRALS],
            p_ge2_target=p_ge2_target,
            seed=int(seed),
        )

    if use_contract and contract_base is not None:
        X_scenario[COL_CONTRACT] = apply_contract_two_year_only(
            X_scenario[COL_CONTRACT],
            p_two_year_target=p_two_year_target,
            seed=int(seed) + 1,
        )

    if use_payment and pay_cc_base is not None and pay_cc_target is not None:
        X_scenario[COL_PAYMENT] = apply_payment_creditcard_target(
            X_scenario[COL_PAYMENT],
            target_cc_rate=pay_cc_target,
            seed=int(seed) + 10,
        )

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
    delta_pp = (scen_rate - base_rate) * 100.0
    delta_color_pp = -delta_pp # invert sign for coloring: decrease -> positive (green), increase -> negative (red)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Baseline churn rate", f"{base_rate:.2%}")
    # feed inverted sign to control color, but show correct text
    delta_text = f"{delta_pp:+.2f} p.p."
    delta_color_text = f"{delta_color_pp:.2f} p.p."  # controls color

    m2.metric(
        f"Scenario churn rate (≥ {threshold:.2f})",
        f"{scen_rate:.2%}",
        delta=delta_color_text,      # drives green/red
        delta_color="normal",
    )

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
        ax.set_ylim(0, 5000)

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
