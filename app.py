import os
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import requests  # Added for downloading data from URL

st.set_page_config(page_title="Property Value Predictor", layout="wide")
st.title("Property Value Predictor (Linear Regression)")
st.write("Loads Hamilton County assessor data, trains a simple Linear Regression model on a few key features, and predicts APPRAISED_VALUE.")

def list_files_here():
    return sorted([p.name for p in Path(".").iterdir() if p.is_file()])

def find_data_file():
    # Prefer ZIP if present
    for p in Path(".").glob("*.zip"):
        if "AssessorExportCSV" in p.name:
            return str(p), None
    # Otherwise any CSV
    for p in Path(".").glob("*.csv"):
        return None, str(p)
    
    # If not found, download from source URL (Hamilton County TN Assessor data)
    url = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"
    zip_path = "AssessorExportCSV.zip"
    st.info(f"Downloading data from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Download complete.")
        return zip_path, None
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return None, None

def unzip_all(zip_path: str, extract_dir: str = "data_assessor") -> list[str]:
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    csv_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".csv") or f.lower().endswith(".txt"):
                csv_files.append(os.path.join(root, f))
    if not csv_files:
        raise FileNotFoundError("No CSV/TXT files found inside the ZIP after extraction.")
    return csv_files

@st.cache_data(show_spinner=True)
def load_df(zip_path: str | None, csv_path: str | None) -> pd.DataFrame:
    if zip_path:
        extracted_files = unzip_all(zip_path)
        # Find main parcel file (largest or by name)
        main_file = max(extracted_files, key=lambda p: os.path.getsize(p))
        df_main = pd.read_csv(main_file, low_memory=False)
        st.caption(f"Loaded main file: {main_file}")
        
        # Find building file (by name or assume second largest)
        building_file = next((f for f in extracted_files if "building" in f.lower()), None)
        if building_file:
            df_building = pd.read_csv(building_file, low_memory=False)
            st.caption(f"Loaded building file: {building_file}")
            # Merge on MAP, GROUP, PARCEL (assume common columns)
            merge_cols = ['MAP', 'GROUP', 'PARCEL']
            if all(c in df_main.columns and c in df_building.columns for c in merge_cols):
                df = pd.merge(df_main, df_building, on=merge_cols, how='left')
            else:
                st.warning("Could not merge building data: missing merge columns.")
                df = df_main
        else:
            st.warning("No building file found; using main file only.")
            df = df_main
        
        df.attrs["source"] = f"ZIP → {main_file} (and building if available)"
        return df
    if csv_path:
        df = pd.read_csv(csv_path, low_memory=False)
        df.attrs["source"] = f"CSV → {csv_path}"
        return df
    raise FileNotFoundError("No AssessorExportCSV.zip or .csv found or downloaded.")

def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    if "APPRAISED_VALUE" not in df.columns:
        raise KeyError("APPRAISED_VALUE column not found in dataset.")
    df = df.copy()
    df["APPRAISED_VALUE"] = (
        df["APPRAISED_VALUE"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
    )
    df["APPRAISED_VALUE"] = pd.to_numeric(df["APPRAISED_VALUE"], errors="coerce")
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0].copy()
    return df

def safe_residential_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    type_cols = ["PROP_TYPE_CODE_DESC", "CURRENT_USE_CODE_DESC", "LAND_USE_CODE_DESC"]
    res_col = next((c for c in type_cols if c in df.columns), None)
    if res_col is None:
        return df
    pattern = r"RES|RESIDENT|SINGLE|DWELL|CONDO|TOWN|APART|MULTI|FAM|HOME|HOUSE|R-1|R-2|R1|R2"
    filtered = df[df[res_col].astype(str).str.contains(pattern, case=False, na=False)].copy()
    # fallback if filter returns 0 rows
    return filtered if len(filtered) > 0 else df

def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["APPRAISED_VALUE"].copy()
    # Define a few key features based on actual column names
    possible_features = ["YearBuilt", "FullBath", "ThreeQuarterBath", "HalfBath", "SizeArea", "CALC_ACRES"]
    selected_features = [c for c in possible_features if c in df.columns]
    if not selected_features:
        raise ValueError("None of the selected key features are available in the dataset.")
    st.caption(f"Using these few features for modeling: {', '.join(selected_features)}")
    
    X_raw = df[selected_features].copy()
    # Convert to numeric where possible
    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    
    X = pd.get_dummies(X_raw, drop_first=True)  # In case any categoricals, but unlikely
    X = X.select_dtypes(include=[np.number, "bool"]).astype(float)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns after processing (X has 0 columns).")
    X = X.fillna(X.median(numeric_only=True))
    X, y = X.align(y, join="inner", axis=0)
    if X.shape[0] < 50:
        raise ValueError(f"Not enough rows to train after cleaning (have {X.shape[0]} rows).")
    return X, y

def train_lr(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mae, r2, list(X.columns)

# ---------- UI: Data status ----------
with st.expander("Data file status", expanded=True):
    st.write("Working directory:", os.getcwd())
    st.write("Files here:", list_files_here())

zip_path, csv_path = find_data_file()
if not zip_path and not csv_path:
    st.error("Unable to find or download AssessorExportCSV.zip or a CSV.")
    st.stop()

# ---------- Load ----------
try:
    df = load_df(zip_path, csv_path)
    st.caption(f"Loaded source: {df.attrs.get('source', 'unknown')}")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ---------- Clean ----------
try:
    df = clean_target(df)
except Exception as e:
    st.error(f"Failed to clean APPRAISED_VALUE: {e}")
    st.stop()

do_res = st.checkbox("Filter to residential properties (safe fallback)", value=True)
df_use = safe_residential_filter(df) if do_res else df
st.write("Rows after APPRAISED_VALUE cleaning:", len(df))
st.write("Rows used for modeling:", len(df_use))
if len(df_use) == 0:
    st.error("0 rows available for modeling. Disable residential filter and retry.")
    st.stop()

# ---------- Build + Train ----------
try:
    X, y = build_xy(df_use)
    model, mae, r2, feature_names = train_lr(X, y)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

st.success("Model trained on selected features.")
st.metric("MAE", f"${mae:,.0f}")
st.metric("R²", f"{r2:.3f}")

# ---------- Predict ----------
st.subheader("Predict APPRAISED_VALUE")
# The few features are the ones used in the model, so allow overrides for all available
preferred_inputs = [c for c in ["YearBuilt", "FullBath", "ThreeQuarterBath", "HalfBath", "SizeArea", "CALC_ACRES"] if c in df_use.columns]
if preferred_inputs:
    st.caption("Enter values for the key features to predict the appraised value.")
    cols = st.columns(2)
    user_raw = {}
    for i, col in enumerate(preferred_inputs):
        default_val = float(pd.to_numeric(df_use[col], errors="coerce").median())
        with cols[i % 2]:
            user_raw[col] = st.number_input(col, value=default_val)
    # Since model uses these exact columns (numeric), create X_user directly
    X_user = pd.DataFrame([user_raw]).reindex(columns=feature_names, fill_value=0).astype(float)
else:
    st.error("No input features available for prediction.")
    st.stop()

pred = float(model.predict(X_user)[0])
st.write("### Predicted APPRAISED_VALUE")
st.write(f"**${pred:,.0f}**")
