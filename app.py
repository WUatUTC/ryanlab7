import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Property Value Predictor", layout="wide")
st.title("Property Value Predictor")
st.write(
    "This app downloads Hamilton County assessor data, merges parcel and building data, "
    "trains a simple Linear Regression model, and predicts APPRAISED_VALUE."
)

PARCEL_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"
BUILDING_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorBuildingExport.zip"


def download_zip_bytes(url: str) -> bytes:
    """Download a ZIP file and return raw bytes."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def read_first_csv_or_txt_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Read the first CSV or TXT file found inside a ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = zf.namelist()
        data_files = [
            name for name in members
            if name.lower().endswith(".csv") or name.lower().endswith(".txt")
        ]
        if not data_files:
            raise FileNotFoundError("No CSV/TXT file found inside ZIP archive.")

        # Prefer CSV if available
        data_files = sorted(data_files, key=lambda x: (not x.lower().endswith(".csv"), x))
        target = data_files[0]

        with zf.open(target) as f:
            if target.lower().endswith(".csv"):
                return pd.read_csv(f, low_memory=False)
            # Fallback for TXT if ever needed
            return pd.read_csv(f, low_memory=False)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces and standardize column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """Download parcel and building exports, then merge them."""
    parcel_zip = download_zip_bytes(PARCEL_ZIP_URL)
    building_zip = download_zip_bytes(BUILDING_ZIP_URL)

    df_parcel = normalize_column_names(read_first_csv_or_txt_from_zip(parcel_zip))
    df_building = normalize_column_names(read_first_csv_or_txt_from_zip(building_zip))

    # Required merge keys
    merge_cols = ["MAP", "GROUP", "PARCEL"]
    missing_parcel = [c for c in merge_cols if c not in df_parcel.columns]
    missing_building = [c for c in merge_cols if c not in df_building.columns]

    if missing_parcel:
        raise KeyError(f"Parcel file missing merge columns: {missing_parcel}")
    if missing_building:
        raise KeyError(f"Building file missing merge columns: {missing_building}")

    # Keep only documented/useful building columns if they exist
    building_keep = merge_cols + [
        "YearBuilt",
        "SizeArea",
        "FullBath",
        "ThreeQuarterBath",
        "HalfBath",
    ]
    building_keep = [c for c in building_keep if c in df_building.columns]

    # If multiple building records exist per parcel, aggregate to one row per parcel
    agg_map = {}
    for col in building_keep:
        if col in merge_cols:
            continue
        agg_map[col] = "max"

    if agg_map:
        df_building_small = (
            df_building[building_keep]
            .groupby(merge_cols, as_index=False)
            .agg(agg_map)
        )
    else:
        df_building_small = df_building[merge_cols].drop_duplicates()

    df = pd.merge(df_parcel, df_building_small, on=merge_cols, how="left")
    return df


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Convert mixed-format numeric text to numeric."""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Clean APPRAISED_VALUE and remove invalid target rows."""
    df = df.copy()
    if "APPRAISED_VALUE" not in df.columns:
        raise KeyError("APPRAISED_VALUE column not found.")

    df["APPRAISED_VALUE"] = clean_numeric_series(df["APPRAISED_VALUE"])
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0].copy()
    return df


def residential_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer a clean property-type filter using the documented parcel descriptions.
    Falls back to the original dataframe if the needed columns are missing.
    """
    df = df.copy()

    # Most direct documented candidate
    if "PROP_TYPE_CODE_DESC" in df.columns:
        mask = df["PROP_TYPE_CODE_DESC"].astype(str).str.contains(
            r"Residential|Apartment|Rental", case=False, na=False
        )
        if mask.any():
            return df[mask].copy()

    # Secondary fallback
    for col in ["LAND_USE_CODE_DESC", "CURRENT_USE_CODE_DESC"]:
        if col in df.columns:
            mask = df[col].astype(str).str.contains(
                r"Residential|Household|Condominium|Apartment|Duplex|Triplex|Home",
                case=False,
                na=False,
            )
            if mask.any():
                return df[mask].copy()

    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build model matrix using only verified/documented feature names.
    """
    df = df.copy()
    y = df["APPRAISED_VALUE"].copy()

    verified_features = [
        "YearBuilt",
        "SizeArea",
        "FullBath",
        "ThreeQuarterBath",
        "HalfBath",
        "CALC_ACRES",
    ]
    selected_features = [c for c in verified_features if c in df.columns]

    if not selected_features:
        raise ValueError(
            "None of the verified predictor columns were found. "
            "Expected one or more of: "
            + ", ".join(verified_features)
        )

    X = df[selected_features].copy()
    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("All selected predictor columns became empty after cleaning.")

    X = X.fillna(X.median(numeric_only=True))

    # Keep aligned rows only
    X, y = X.align(y, join="inner", axis=0)

    if len(X) < 50:
        raise ValueError(f"Not enough rows after cleaning: {len(X)}")

    return X.astype(float), y.astype(float), selected_features


def train_model(X: pd.DataFrame, y: pd.Series):
    """Train/test split and fit linear regression."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2


# ----------------------------
# Load + clean
# ----------------------------
try:
    with st.spinner("Downloading and loading parcel/building data..."):
        df = load_data()
    st.success("Data loaded successfully.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.write("Rows before cleaning:", len(df))
st.write("Columns found:", len(df.columns))

try:
    df = clean_target(df)
except Exception as e:
    st.error(f"Failed to clean target column: {e}")
    st.stop()

use_residential = st.checkbox("Filter to residential-like properties", value=True)
df_use = residential_filter(df) if use_residential else df

st.write("Rows after target cleaning:", len(df))
st.write("Rows used for modeling:", len(df_use))

if len(df_use) == 0:
    st.error("No rows available for modeling.")
    st.stop()

# ----------------------------
# Train
# ----------------------------
try:
    X, y, feature_names = build_xy(df_use)
    model, mae, r2 = train_model(X, y)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

st.success("Model trained successfully.")
c1, c2 = st.columns(2)
c1.metric("MAE", f"${mae:,.0f}")
c2.metric("R²", f"{r2:.3f}")

st.subheader("Features used")
st.write(feature_names)

# ----------------------------
# Predict
# ----------------------------
st.subheader("Predict APPRAISED_VALUE")
st.caption("Enter values for the same features used in training.")

user_input = {}
cols = st.columns(2)
for i, col in enumerate(feature_names):
    default_val = float(pd.to_numeric(df_use[col], errors="coerce").median())
    with cols[i % 2]:
        user_input[col] = st.number_input(col, value=default_val)

X_user = pd.DataFrame([user_input])[feature_names].astype(float)
pred = float(model.predict(X_user)[0])

st.write("### Predicted APPRAISED_VALUE")
st.write(f"**${pred:,.0f}**")

# ----------------------------
# Optional preview
# ----------------------------
with st.expander("Preview cleaned data"):
    st.dataframe(df_use[feature_names + ["APPRAISED_VALUE"]].head(20))
