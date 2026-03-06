# app.py
# Streamlit app to train a linear model (Ridge) for APPRAISED_VALUE from assessor exports.
# - Supports loading data from a URL (ZIP/CSV) or local files.
# - Uses a leakage-safe Pipeline with ColumnTransformer and OneHotEncoder.
# - Optional log-target training for improved MAE on skewed prices.

import os
import zipfile
import hashlib
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import requests
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

# ---------------------------- Streamlit setup ----------------------------
st.set_page_config(page_title="Lab 7 - Property Value Predictor", layout="wide")
st.title("Lab 7: Property Value Predictor (Linear Regression)")
st.write(
    "Loads assessor export (ZIP/CSV) from URL or local folder, trains a regularized linear model, "
    "and predicts **APPRAISED_VALUE**."
)

# ---------------------------- File utilities ----------------------------
def list_files_here() -> List[str]:
    return sorted([p.name for p in Path(".").iterdir() if p.is_file()])


def _mtime(path: Optional[str]) -> Optional[float]:
    return os.path.getmtime(path) if path and os.path.exists(path) else None


def find_data_file() -> Tuple[Optional[str], Optional[str]]:
    """
    Prefer ZIP named with 'AssessorExportCSV' if present; otherwise pick the largest CSV in folder.
    Returns (zip_path, csv_path)
    """
    zips = [p for p in Path(".").glob("*.zip") if "AssessorExportCSV" in p.name]
    if zips:
        zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return str(zips[0]), None
    csvs = list(Path(".").glob("*.csv"))
    if csvs:
        csvs.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return None, str(csvs[0])
    return None, None


def unzip_and_pick_csv(zip_path: str, extract_dir: str = "data_assessor") -> str:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    csv_files: List[str] = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise FileNotFoundError("No CSV files found inside the ZIP after extraction.")

    # pick largest CSV (typically the main export)
    return max(csv_files, key=lambda p: os.path.getsize(p))


# ---------------------------- URL download helpers ----------------------------
def _url_fingerprint(url: str, headers: Dict[str, str]) -> str:
    """
    Build a cache key using URL + ETag/Last-Modified if provided by server.
    Falls back to URL-only hash if not present.
    """
    etag = headers.get("ETag") or headers.get("Etag") or ""
    last_mod = headers.get("Last-Modified") or headers.get("last-modified") or ""
    raw = f"{url}|{etag}|{last_mod}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@st.cache_data(show_spinner=True)
def download_to_cache(url: str, key: str) -> str:
    """
    Download the URL to a temp file and return its path. Cached by 'key'.
    Detects a suitable file extension (.zip or .csv) for later steps.
    """
    resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    cd = resp.headers.get("Content-Disposition", "")
    filename = None
    if "filename=" in cd:
        # Basic parse of filename=...; strip quotes/semicolons
        filename = cd.split("filename=")[-1].strip('"; ')

    if filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
    else:
        if "zip" in content_type or url.lower().endswith(".zip"):
            ext = ".zip"
        elif "csv" in content_type or url.lower().endswith(".csv"):
            ext = ".csv"
        else:
            guess = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
            ext = guess if guess in (".zip", ".csv") else ".csv"

    tmp_dir = Path(tempfile.gettempdir()) / "lab7_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{key}{ext}"

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return str(out_path)


def fetch_remote_file(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (zip_path, csv_path) based on the remote file.
    - If remote is ZIP: (zip_path, None)
    - If remote is CSV: (None, csv_path)
    """
    try:
        head = requests.head(url, timeout=15, allow_redirects=True)
        head.raise_for_status()
        fp = _url_fingerprint(url, head.headers)
    except Exception:
        fp = hashlib.sha256(url.encode("utf-8")).hexdigest()

    path = download_to_cache(url, fp)
    if path.lower().endswith(".zip"):
        return path, None
    elif path.lower().endswith(".csv"):
        return None, path
    # default to CSV if unknown
    return None, path


# ---------------------------- Load & clean dataset ----------------------------
@st.cache_data(show_spinner=True)
def load_df(zip_path: Optional[str], csv_path: Optional[str],
            zip_mtime: Optional[float], csv_mtime: Optional[float]) -> pd.DataFrame:
    """
    Load a CSV either from a ZIP or a direct CSV path.
    Cache is invalidated when the file modified time changes.
    """
    if zip_path:
        csv_extracted = unzip_and_pick_csv(zip_path)
        df = pd.read_csv(csv_extracted, low_memory=False)
        df.attrs["source"] = f"ZIP → {csv_extracted}"
        return df
    if csv_path:
        df = pd.read_csv(csv_path, low_memory=False)
        df.attrs["source"] = f"CSV → {csv_path}"
        return df
    raise FileNotFoundError("No AssessorExportCSV.zip or .csv found.")


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
    """
    Try to focus on residential using whichever of these columns exists.
    Falls back to full df if filter yields 0 rows.
    """
    df = df.copy()
    type_cols = ["PROPERTY_CLASS", "CURRENT_USE_CODE_DESC", "CURRENT_USE_CODE"]
    res_col = next((c for c in type_cols if c in df.columns), None)
    if res_col is None:
        return df
    pattern = r"RES|RESIDENT|SINGLE|DWELL|CONDO|TOWN|APART|MULTI|FAM|HOME|HOUSE|R-1|R-2|R1|R2"
    filtered = df[df[res_col].astype(str).str.contains(pattern, case=False, na=False)].copy()
    return filtered if len(filtered) > 0 else df


def drop_id_like_and_high_cardinality(df: pd.DataFrame,
                                      max_unique_ratio: float = 0.5,
                                      max_unique_abs: int = 500) -> pd.DataFrame:
    """
    Remove identifiers and ultra high-cardinality text columns (e.g., parcel IDs, full addresses).
    """
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


# ---------------------------- Modeling helpers ----------------------------
def build_pipeline(X_raw: pd.DataFrame) -> Pipeline:
    """
    Build a leakage-safe sklearn Pipeline:
      - SimpleImputer(median) for numeric
      - SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore") for categorical
      - Ridge() as a stable linear model
    """
    num_cols = X_raw.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X_raw.columns if c not in num_cols]

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
    return pipe, num_cols, cat_cols


def make_single_row_defaults(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row DataFrame containing defaults for ALL columns in X_raw:
      - numeric/bool: median (coerced to float for numeric)
      - categorical: most frequent value or empty string
    This ensures ColumnTransformer selectors find every column at predict time.
    """
    defaults: Dict[str, object] = {}
    for c in X_raw.columns:
        s = X_raw[c]
        if s.dtype.kind in "biufc":
            defaults[c] = float(pd.to_numeric(s, errors="coerce").median())
        elif s.dtype == bool:
            mode = s.mode(dropna=True)
            defaults[c] = bool(mode.iloc[0]) if not mode.empty else False
        else:
            mode = s.astype(str).mode(dropna=True)
            defaults[c] = mode.iloc[0] if not mode.empty else ""
    return pd.DataFrame([defaults])


# ---------------------------- UI: Data source ----------------------------
with st.expander("Data file status", expanded=True):
    st.write("Working directory:", os.getcwd())
    st.write("Files here:", list_files_here())

st.subheader("Data source")

default_url = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"
if "data_url" not in st.session_state:
    st.session_state["data_url"] = default_url

remote_url = st.text_input(
    "Provide a CSV/ZIP URL (optional):",
    value=st.session_state["data_url"],
    help="Paste a direct link to AssessorExportCSV.zip or a CSV."
)
st.session_state["data_url"] = remote_url

zip_path: Optional[str] = None
csv_path: Optional[str] = None

if remote_url.strip():
    try:
        zip_path, csv_path = fetch_remote_file(remote_url.strip())
        st.caption(f"Fetched from URL → {'ZIP' if zip_path else 'CSV'}")
    except Exception as e:
        st.error(f"Failed to fetch from URL: {e}")
        st.stop()
else:
    z, c = find_data_file()
    zip_path, csv_path = z, c

if not zip_path and not csv_path:
    st.error("No data file found. Provide a URL above or place AssessorExportCSV.zip (or a CSV) in this folder, then rerun.")
    st.stop()

# ---------------------------- Load ----------------------------
try:
    df = load_df(zip_path, csv_path, _mtime(zip_path), _mtime(csv_path))
    src = "URL" if remote_url.strip() else "local"
    st.caption(f"Loaded source ({src}): {df.attrs.get('source', 'unknown')}")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ---------------------------- Clean target ----------------------------
try:
    df = clean_target(df)
except Exception as e:
    st.error(f"Failed to clean APPRAISED_VALUE: {e}")
    st.stop()

# ---------------------------- Optional residential filter ----------------------------
do_res = st.checkbox("Filter to residential properties (safe fallback)", value=True)
df_use = safe_residential_filter(df) if do_res else df

st.write("Rows after APPRAISED_VALUE cleaning:", len(df))
st.write("Rows used for modeling:", len(df_use))
if len(df_use) == 0:
    st.error("0 rows available for modeling. Disable residential filter and retry.")
    st.stop()

# ---------------------------- Drop ID-like/high-cardinality ----------------------------
df_model = drop_id_like_and_high_cardinality(df_use)

# ---------------------------- Prepare raw X/y ----------------------------
target_col = "APPRAISED_VALUE"
drop_cols = [c for c in [target_col, "OWNER_NAME_1", "OWNER_NAME_2", "OWNER_NAME_3", "GISLINK"] if c in df_model.columns]
X_raw = df_model.drop(columns=drop_cols, errors="ignore")
y = df_model[target_col].copy()

if X_raw.shape[0] < 50:
    st.error(f"Not enough rows to train after cleaning (have {X_raw.shape[0]} rows).")
    st.stop()

# ---------------------------- Build pipeline & train ----------------------------
pipe, num_cols, cat_cols = build_pipeline(X_raw)
use_log_target = True  # toggle: train on log1p(y) and expm1 back to dollars

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42)

if use_log_target:
    pipe.fit(X_train, np.log1p(y_train))
    y_pred = np.expm1(pipe.predict(X_test))
else:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
baseline_mae = (y_test - y_train.median()).abs().mean()  # naive baseline

st.success("Model trained.")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("MAE", f"${mae:,.0f}")
col_m2.metric("R²", f"{r2:.3f}")
col_m3.caption(f"Baseline (predict train median): MAE ${baseline_mae:,.0f}")

# ---------------------------- Cross-validation (optional) ----------------------------
with st.expander("Cross-validation (5-fold)", expanded=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    if use_log_target:
        # custom CV loop to score in dollars when fitting in log-space
        def cv_mae(pipe_in: Pipeline, X: pd.DataFrame, y_: pd.Series) -> np.ndarray:
            scores = []
            for tr, te in kf.split(X):
                Xtr, Xte = X.iloc[tr], X.iloc[te]
                ytr, yte = y_.iloc[tr], y_.iloc[te]
                pipe_in.fit(Xtr, np.log1p(ytr))
                yp = np.expm1(pipe_in.predict(Xte))
                scores.append(mean_absolute_error(yte, yp))
            return np.array(scores)

        cv_mae_scores = cv_mae(pipe, X_raw, y)
        st.write(f"CV MAE: ${cv_mae_scores.mean():,.0f} ± ${cv_mae_scores.std():,.0f}")
    else:
        cv_mae_scores = -cross_val_score(pipe, X_raw, y, scoring="neg_mean_absolute_error", cv=kf)
        st.write(f"CV MAE: ${cv_mae_scores.mean():,.0f} ± ${cv_mae_scores.std():,.0f}")

# ---------------------------- Predict ----------------------------
st.subheader("Predict APPRAISED_VALUE")

# Build a full default row for ALL columns so ColumnTransformer finds every column
X_user_raw = make_single_row_defaults(X_raw)

# Let users override a few common numeric inputs if present
preferred_inputs = [c for c in ["YEAR_BUILT", "BEDROOMS", "FULL_BATH", "HALF_BATH",
                                "TOTAL_LIVING_AREA", "LOT_SIZE", "ACRES"]
                    if c in X_user_raw.columns]

if preferred_inputs:
    st.caption("Optional: override a few common fields (if present in your dataset).")
    cols = st.columns(2)
    for i, col in enumerate(preferred_inputs):
        default_val = float(X_user_raw.loc[0, col])
        with cols[i % 2]:
            X_user_raw.loc[0, col] = st.number_input(col, value=default_val)

# Predict via pipeline
if use_log_target:
    pred_val = float(np.expm1(pipe.predict(X_user_raw)[0]))
else:
    pred_val = float(pipe.predict(X_user_raw)[0])

st.write("### Predicted APPRAISED_VALUE")
st.write(f"**${pred_val:,.0f}**")

