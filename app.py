import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# CONFIG 
# ----------------------------

APP_TITLE = "Churn Prediction Dashboard"

MODEL_PATH = "models/best_lgb_model.pkl"

DF_FINAL_PATH = "data/final_dfs/df_final.parquet"  
X_PATH = "data/final_dfs/X.parquet"                

# Identify customers 
CUSTOMER_ID_COL = "Customer ID" 

# Columns in df_final containing baseline prediction/prob and customer segment
BASELINE_PROBA_COL = "Churn Prediction Probability"   
BASELINE_PRED_COL = "Churn Prediction"    
SEGMENT_COL = "segment"             

# Features to Simulate
WHATIF_NUMERIC_FEATURES = [
    # Empty for now
]

WHATIF_CATEGORICAL_FEATURES = [
    "Contract Duration",
    "Number of Referrals_bins",
    "Internet Type"
]

# Mapping from a df_final categorical feature to one-hot columns in X
# Example:
#   "contract_type": {
#        "Month-to-month": ["contract_type_Month-to-month"],
#        "One year": ["contract_type_One year"],
#        "Two year": ["contract_type_Two year"],
#   }
ONEHOT_MAP = {
    # "contract_type": {
    #     "Month-to-month": ["contract_type_Month-to-month"],
    #     "One year": ["contract_type_One year"],
    #     "Two year": ["contract_type_Two year"],
    # }
}

# For numeric features, specify which column in X corresponds to that feature.
# If you used a simple scaler without renaming, it's often the same name.
# Example: {"tenure_months": "tenure_months"}
NUMERIC_X_COL_MAP = {
    # "monthly_charges": "monthly_charges",
    # "tenure_months": "tenure_months",
}

APP_TITLE = "Churn Prediction Dashboard"


# ----------------------------
# HELPERS
# ----------------------------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Unsupported file type. Use .parquet or .csv")

@st.cache_data
def build_robust_scaler_stats(df_final: pd.DataFrame, numeric_features):
    """
    Compute RobustScaler parameters from df_final: median, q1, q3, iqr.
    NOTE: Ideally compute from the TRAIN set used during training.
    """
    stats = {}
    for f in numeric_features:
        if f not in df_final.columns:
            raise KeyError(f"Numeric feature '{f}' not found in df_final.")

        s = pd.to_numeric(df_final[f], errors="coerce").dropna()
        if s.empty:
            raise ValueError(f"Numeric feature '{f}' has no valid numeric values.")

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        med = float(s.median())
        iqr = q3 - q1

        # Guard: avoid divide-by-zero (constant feature)
        if iqr == 0:
            iqr = 1e-9

        stats[f] = {"median": med, "iqr": iqr, "q1": q1, "q3": q3}
    return stats

def robust_scale(value: float, median: float, iqr: float) -> float:
    return float((value - median) / iqr)


def get_customer_key(df_final: pd.DataFrame, row_idx: int):
    if CUSTOMER_ID_COL and CUSTOMER_ID_COL in df_final.columns:
        return df_final.iloc[row_idx][CUSTOMER_ID_COL]
    return row_idx


def predict_proba(model, x_row: pd.Series) -> float:
    # LightGBM sklearn wrapper uses predict_proba
    proba = model.predict_proba(x_row.to_frame().T)[:, 1]
    return float(proba[0])


# ----------------------------
# APP
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Load assets
try:
    model = load_model(MODEL_PATH)
    df_final = load_table(DF_FINAL_PATH)
    X = load_table(X_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

# Basic validation
if len(df_final) != len(X):
    st.error(f"df_final and X must have the same number of rows. Got {len(df_final)} vs {len(X)}.")
    st.stop()

# Build numeric mapping functions (original -> scaled)
try:
    numeric_interpolators = build_numeric_interpolators(
        df_final, X,
        WHATIF_NUMERIC_FEATURES,
        NUMERIC_X_COL_MAP
    )
except Exception as e:
    st.warning("Numeric what-if mapping not ready yet. Fix config at top of app.py.")
    st.warning(str(e))
    numeric_interpolators = {}

tab1, tab2 = st.tabs(["Tab 1 — What-if Simulator", "Tab 2 — Model & Insights"])


# ----------------------------
# TAB 1: WHAT-IF
# ----------------------------
with tab1:
    st.subheader("What-if Simulator")

    left, right = st.columns([1, 2])

    with left:
        # Customer selection
        if CUSTOMER_ID_COL and CUSTOMER_ID_COL in df_final.columns:
            customer_ids = df_final[CUSTOMER_ID_COL].astype(str).tolist()
            selected_id = st.selectbox("Select customer", customer_ids)
            idx = df_final.index[df_final[CUSTOMER_ID_COL].astype(str) == selected_id][0]
        else:
            idx = st.number_input("Select row index", min_value=0, max_value=len(df_final)-1, value=0, step=1)

        baseline_x = X.iloc[int(idx)].copy()
        baseline_proba = predict_proba(model, baseline_x)

        st.markdown("### Baseline")
        st.write("Customer:", get_customer_key(df_final, int(idx)))
        if SEGMENT_COL and SEGMENT_COL in df_final.columns:
            st.write("Segment:", df_final.iloc[int(idx)][SEGMENT_COL])

        st.metric("Baseline churn probability", f"{baseline_proba:.3f}")

        if BASELINE_PROBA_COL and BASELINE_PROBA_COL in df_final.columns:
            st.caption(f"(df_final stored baseline proba: {df_final.iloc[int(idx)][BASELINE_PROBA_COL]})")

    with right:
        st.markdown("### Edit features (human-readable)")

        # Start from baseline X row (scaled/encoded)
        new_x = baseline_x.copy()

        # Numeric edits
        if WHATIF_NUMERIC_FEATURES:
            st.markdown("#### Numeric features")
            for f in WHATIF_NUMERIC_FEATURES:
                if f not in df_final.columns:
                    st.warning(f"'{f}' not found in df_final")
                    continue
                current_val = df_final.iloc[int(idx)][f]

                # slider range from quantiles for stability
                q01 = float(df_final[f].quantile(0.01))
                q99 = float(df_final[f].quantile(0.99))
                if not np.isfinite(q01) or not np.isfinite(q99) or q01 == q99:
                    q01 = float(df_final[f].min())
                    q99 = float(df_final[f].max())

                v_new = st.slider(
                    label=f,
                    min_value=float(q01),
                    max_value=float(q99),
                    value=float(np.clip(current_val, q01, q99)) if np.isfinite(current_val) else float(q01),
                )

                # Map to scaled column in X via interpolator
                if f in numeric_interpolators and f in NUMERIC_X_COL_MAP:
                    x_col = NUMERIC_X_COL_MAP[f]
                    new_x[x_col] = numeric_interpolators[f](v_new)
                else:
                    st.info(f"No mapping for '{f}'. Configure NUMERIC_X_COL_MAP and ensure interpolator builds.")

        # Categorical edits (one-hot)
        if WHATIF_CATEGORICAL_FEATURES:
            st.markdown("#### Categorical features")
            for f in WHATIF_CATEGORICAL_FEATURES:
                if f not in df_final.columns:
                    st.warning(f"'{f}' not found in df_final")
                    continue
                if f not in ONEHOT_MAP:
                    st.info(f"No one-hot mapping for '{f}'. Configure ONEHOT_MAP.")
                    continue

                options = list(ONEHOT_MAP[f].keys())
                current = df_final.iloc[int(idx)][f]
                default_idx = options.index(current) if current in options else 0

                chosen = st.selectbox(f, options, index=default_idx)

                # Zero all relevant one-hot columns
                all_cols = sorted({c for cols in ONEHOT_MAP[f].values() for c in cols})
                for c in all_cols:
                    if c in new_x.index:
                        new_x[c] = 0.0

                # Set chosen category columns to 1
                for c in ONEHOT_MAP[f][chosen]:
                    if c in new_x.index:
                        new_x[c] = 1.0

        st.divider()

        # Predict with modified vector
        new_proba = predict_proba(model, new_x)
        delta = new_proba - baseline_proba

        c1, c2, c3 = st.columns(3)
        c1.metric("New churn probability", f"{new_proba:.3f}", f"{delta:+.3f}")
        c2.metric("Baseline", f"{baseline_proba:.3f}")
        c3.metric("Change", f"{delta:+.3f}")

        with st.expander("Show edited X row (model input)"):
            st.dataframe(pd.DataFrame({"baseline": baseline_x, "new": new_x}))


# ----------------------------
# TAB 2: MODEL & INSIGHTS
# ----------------------------
with tab2:
    st.subheader("Model & Insights")

    st.markdown("### Model summary")
    st.write("Model type:", type(model).__name__)
    st.write("Number of features:", X.shape[1])
    st.write("Number of customers:", X.shape[0])

    with st.expander("Model parameters"):
        try:
            st.json(model.get_params())
        except Exception:
            st.write("Could not read model params via get_params().")

    st.markdown("### Feature importance (gain-based / split-based depending on wrapper)")
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            st.info("No feature_importances_ found on this model object.")
        else:
            fi = pd.DataFrame({
                "feature": X.columns,
                "importance": importances
            }).sort_values("importance", ascending=False)

            st.dataframe(fi.head(30), use_container_width=True)

            # Simple bar chart
            st.bar_chart(fi.head(30).set_index("feature")["importance"])
    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")

    st.markdown("### Baseline churn distribution")
    # Show distribution of model probabilities for all customers
    try:
        # predict in chunks if needed later; for now assume manageable
        probs = model.predict_proba(X)[:, 1]
        s = pd.Series(probs, name="churn_probability")
        st.write(s.describe())
        st.line_chart(s.sort_values(ignore_index=True))
    except Exception as e:
        st.warning(f"Could not compute probability distribution: {e}")

    st.markdown("### Your model results")
    st.info(
        "If you have a metrics JSON (AUC/F1/confusion matrix) from training, "
        "save it as `models/metrics.json` and we can load & display it here."
    )
