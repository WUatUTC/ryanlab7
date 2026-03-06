import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Lab 7 - Property Value Predictor", layout="wide")
st.title("Lab 7: Property Value Predictor (Linear Regression)")
st.write("Loads assessor export (ZIP/CSV), trains a regularized linear model, predicts APPRAISED_VALUE.")

# ---------------------------- Utilities ----------------------------
def list_files_here():
    return sorted([p.name for p in Path(".").iterdir() if p.is_file()])

def _mtime(path):
    return os.path.getmtime(path) if path and os.path.exists(path) else None

def find_data_file():
    # Prefer ZIP if present
    zips = [p for p in Path(".").glob("*.zip") if "AssessorExportCSV" in p.name]
    if zips:
        # If multiple, choose the newest by mtime
        zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return str(zips[0]), None
    # Otherwise pick the largest CSV in folder (consistent with ZIP heuristic)
    csvs = list(Path(".").glob("*.csv"))
    if csvs:
        csvs.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return None, str(csvs[0])
    return None, None

def unzip_and_pick_csv(zip_path: str, extract_dir: str = "data_assessor") -> str:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    csv_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    if not csv_files:
        raise FileNotFoundError("No CSV files found inside the ZIP after extraction.")
    # pick largest CSV (typically the main export)
    return max(csv_files, key=lambda p: os.path.getsize(p))

@st.cache_data(show_spinner=True)
def load_df(zip_path: str | None, csv_path: str | None, zip_mtime, csv_mtime) -> pd.DataFrame:
    if zip_path:
        csv_extracted = unzip_and_pick_csv(zip_path)
        df = pd.read_csv(csv_extracted, low_memory=False)
        df.attrs["source"] = f"ZIP → {csv_extracted}"
        return df
    if csv_path:
        df = pd.read_csv(csv_path, low_memory=False)
        df.attrs["source"] = f"CSV → {csv_path}"
        return df
    raise FileNotFoundError("No AssessorExportCSV.zip or .csv found in the current folder.")

def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    if "APPRAISED_VALUE" not in df.columns:
        raise KeyError("APPRAISED_VALUE column not found in dataset.")
    df = df.copy()
    df["APPRAISED_VALUE"] = (
        df["APPRAISED_VALUE"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
    )
    df["APPRAISED_VALUE"] = pd.to_numeric(df["APPRAISED_VALUE"], errors="coerce")
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0].copy()
    return df

def safe_residential_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    type_cols = ["PROPERTY_CLASS", "CURRENT_USE_CODE_DESC", "CURRENT_USE_CODE"]
    res_col = next((c for c in type_cols if c in df.columns), None)
    if res_col is None:  # no column to filter with
        return df
    pattern = r"RES|RESIDENT|SINGLE|DWELL|CONDO|TOWN|APART|MULTI|FAM|HOME|HOUSE|R-1|R-2|R1|R2"
    filtered = df[df[res_col].astype(str).str.contains(pattern, case=False, na=False)].copy()
    return filtered if len(filtered) > 0 else df  # fallback

def drop_id_like_and_high_cardinality(df, max_unique_ratio=0.5, max_unique_abs=500):
    """Remove identifiers and text-y / ultra high-cardinality categorical features."""
    df = df.copy()
    bad_patterns = ["PARCEL", "ACCOUNT", "PIN", "APN", "FOLIO", "MAP", "ADDR", "ADDRESS", "GISLINK", "ID", "LINK"]
    to_drop = set()
    for c in df.columns:
        cu = c.upper()
        if any(pat in cu for pat in bad_patterns):
            to_drop.add(c)
        elif df[c].dtype == "object":
            nun = df[c].nunique(dropna=True)
            if nun > max_unique_abs and nun > max_unique_ratio * len(df):
                to_drop.add(c)
    return df.drop(columns=list(to_drop), errors="ignore")

# ---------------------------- UI: Data status ----------------------------
with st.expander("Data file status", expanded=True):
    st.write("Working directory:", os.getcwd())
    st.write("Files here:", list_files_here())

zip_path, csv_path = find_data_file()

if not zip_path and not csv_path:
    st.error("Place AssessorExportCSV.zip (or a CSV) in this folder, then rerun.")
    st.stop()

# ---------- Load ----------
try:
    df = load_df(zip_path, csv_path, _mtime(zip_path), _mtime(csv_path))
    st.caption(f"Loaded source: {df.attrs.get('source', 'unknown')}")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ---------- Clean target ----------
try:
    df = clean_target(df)
except Exception as e:
    st.error(f"Failed to clean APPRAISED_VALUE: {e}")
    st.stop()

# ---------- Optional residential filter ----------
do_res = st.checkbox("Filter to residential properties (safe fallback)", value=True)
df_use = safe_residential_filter(df) if do_res else df

st.write("Rows after APPRAISED_VALUE cleaning:", len(df))
st.write("Rows used for modeling:", len(df_use))
if len(df_use) == 0:
    st.error("0 rows available for modeling. Disable residential filter and retry.")
    st.stop()

# ---------- Drop ID-like / very high-card columns ----------
df_model = drop_id_like_and_high_cardinality(df_use)

# ---------- Prepare raw X/y for pipeline ----------
target_col = "APPRAISED_VALUE"
drop_cols = [c for c in [target_col, "OWNER_NAME_1", "OWNER_NAME_2", "OWNER_NAME_3", "GISLINK"] if c in df_model.columns]
X_raw = df_model.drop(columns=drop_cols, errors="ignore")
y = df_model[target_col].copy()

# Column types for transformer
num_cols = X_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = [c for c in X_raw.columns if c not in num_cols]

# ---------- Build pipeline (impute + OHE + Ridge) ----------
pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ],
    remainder="drop"
)

model = Ridge(alpha=1.0, random_state=42)
pipe = Pipeline(steps=[("pre", pre), ("model", model)])

# Option: log target for stability
use_log_target = True

# ---------- Train/test split ----------
if X_raw.shape[0] < 50:
    st.error(f"Not enough rows to train after cleaning (have {X_raw.shape[0]} rows).")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42)

# ---------- Fit ----------
if use_log_target:
    pipe.fit(X_train, np.log1p(y_train))
    y_pred = np.expm1(pipe.predict(X_test))
else:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

# ---------- Metrics ----------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
baseline_mae = (y_test - y_train.median()).abs().mean()  # simple baseline: predict median of train

st.success("Model trained.")
st.metric("MAE", f"${mae:,.0f}")
st.metric("R²", f"{r2:.3f}")
st.caption(f"Baseline (predict train median): MAE ${baseline_mae:,.0f}")

# Optional: 5-fold CV (expensive on very wide data)
with st.expander("Cross-validation (5-fold)", expanded=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    if use_log_target:
        # Wrap target transform for CV via a small helper
        # (fit on log1p, score on dollars)
        def cv_mae(pipe, X, y):
            scores = []
            for tr, te in kf.split(X):
                Xtr, Xte = X.iloc[tr], X.iloc[te]
                ytr, yte = y.iloc[tr], y.iloc[te]
                pipe.fit(Xtr, np.log1p(ytr))
                yp = np.expm1(pipe.predict(Xte))
                scores.append(mean_absolute_error(yte, yp))
            return np.array(scores)
        mae_scores = cv_mae(pipe, X_raw, y)
        st.write(f"CV MAE: ${mae_scores.mean():,.0f} ± ${mae_scores.std():,.0f}")
    else:
        mae_scores = -cross_val_score(pipe, X_raw, y, scoring="neg_mean_absolute_error", cv=kf)
        st.write(f"CV MAE: ${mae_scores.mean():,.0f} ± ${mae_scores.std():,.0f}")

# ---------- Predict ----------
st.subheader("Predict APPRAISED_VALUE")

preferred_inputs = [c for c in ["YEAR_BUILT", "BEDROOMS", "FULL_BATH", "HALF_BATH",
                                "TOTAL_LIVING_AREA", "LOT_SIZE", "ACRES"]
                    if c in X_raw.columns]

# defaults from medians on raw data (numeric coercion where needed)
defaults = {}
for c in preferred_inputs:
    defaults[c] = float(pd.to_numeric(X_raw[c], errors="coerce").median())

X_user_raw = pd.DataFrame([defaults]) if defaults else pd.DataFrame([{}])

if preferred_inputs:
    st.caption("Optional: override a few common fields (if present in your dataset).")
    cols = st.columns(2)
    for i, col in enumerate(preferred_inputs):
        with cols[i % 2]:
            X_user_raw.loc[0, col] = st.number_input(col, value=float(X_user_raw.loc[0, col]))

# For any missing raw columns, fill with simple defaults (median for numeric; most frequent for cat)
# but since we have a pipeline with imputers, we can pass sparse raw data safely.
if use_log_target:
    pred_val = float(np.expm1(pipe.predict(X_user_raw)[0]))
else:
    pred_val = float(pipe.predict(X_user_raw)[0])

st.write("### Predicted APPRAISED_VALUE")
st.write(f"**${pred_val:,.0f}**")
