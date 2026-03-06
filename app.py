import io
import zipfile

import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Property Value Predictor", layout="wide")
st.title("Hamilton County Property Value Predictor")
st.write(
    "This app downloads parcel and building data from Hamilton County, "
    "trains a linear regression model, and predicts appraised property value."
)

PARCEL_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"
BUILDING_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorBuildingExport.zip"


def download_zip_bytes(url: str) -> bytes:
    """Download ZIP file from URL and return raw bytes."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Convert mixed text/numeric values to numeric."""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )


def read_parcel_csv_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Read parcel CSV from ZIP."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in parcel ZIP.")

        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    return normalize_cols(df)


def read_building_file_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    """
    Read building export from ZIP.
    Try CSV first. If TXT, parse as fixed-width text.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names_in_zip = zf.namelist()
        csv_files = [n for n in names_in_zip if n.lower().endswith(".csv")]
        txt_files = [n for n in names_in_zip if n.lower().endswith(".txt")]

        if csv_files:
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f, low_memory=False)
            return normalize_cols(df)

        if txt_files:
            with zf.open(txt_files[0]) as f:
                # Based on the county building extract layout
                colspecs = [
                    (0, 12),      # MAP
                    (12, 24),     # GROUP
                    (24, 36),     # PARCEL
                    (156, 160),   # YEARBUILT
                    (180, 200),   # SIZEAREA
                    (945, 949),   # FULLBATH
                    (949, 953),   # THREEQUARTERBATH
                    (953, 957),   # HALFBATH
                ]
                names = [
                    "MAP",
                    "GROUP",
                    "PARCEL",
                    "YEARBUILT",
                    "SIZEAREA",
                    "FULLBATH",
                    "THREEQUARTERBATH",
                    "HALFBATH",
                ]
                df = pd.read_fwf(f, colspecs=colspecs, names=names, dtype=str)

            return normalize_cols(df)

        raise FileNotFoundError("No CSV or TXT file found in building ZIP.")


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """Download and merge parcel + building data."""
    parcel_zip = download_zip_bytes(PARCEL_ZIP_URL)
    building_zip = download_zip_bytes(BUILDING_ZIP_URL)

    df_parcel = read_parcel_csv_from_zip(parcel_zip)
    df_building = read_building_file_from_zip(building_zip)

    merge_cols = ["MAP", "GROUP", "PARCEL"]

    missing_parcel = [c for c in merge_cols if c not in df_parcel.columns]
    missing_building = [c for c in merge_cols if c not in df_building.columns]

    if missing_parcel:
        raise KeyError(f"Parcel file missing merge columns: {missing_parcel}")
    if missing_building:
        raise KeyError(f"Building file missing merge columns: {missing_building}")

    keep_cols = merge_cols + [
        "YEARBUILT",
        "SIZEAREA",
        "FULLBATH",
        "THREEQUARTERBATH",
        "HALFBATH",
    ]
    keep_cols = [c for c in keep_cols if c in df_building.columns]

    df_building = df_building[keep_cols].copy()

    # Some parcels may have multiple building records
    agg_map = {c: "max" for c in keep_cols if c not in merge_cols}
    df_building = df_building.groupby(merge_cols, as_index=False).agg(agg_map)

    df = pd.merge(df_parcel, df_building, on=merge_cols, how="left")
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Clean APPRAISED_VALUE."""
    df = df.copy()
    if "APPRAISED_VALUE" not in df.columns:
        raise KeyError("APPRAISED_VALUE column not found.")

    df["APPRAISED_VALUE"] = clean_numeric_series(df["APPRAISED_VALUE"])
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0].copy()
    return df


def residential_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to residential-like properties using county coded fields first,
    then description fields as fallback.
    """
    df = df.copy()

    # 1) Property type code
    if "PROP_TYPE_CODE" in df.columns:
        prop_code = pd.to_numeric(df["PROP_TYPE_CODE"], errors="coerce")
        mask = prop_code.isin([22, 32, 40])
        if mask.any():
            return df[mask].copy()

    # 2) Land use code
    if "LAND_USE_CODE" in df.columns:
        lu = pd.to_numeric(df["LAND_USE_CODE"], errors="coerce")
        mask = ((lu >= 100) & (lu < 200)) | lu.isin([111, 112, 113, 114, 115, 116, 117])
        if mask.any():
            return df[mask].copy()

    # 3) Description fallback
    desc_cols = [
        "PROP_TYPE_CODE_DESC",
        "LAND_USE_CODE_DESC",
        "CURRENT_USE_CODE_DESC",
    ]
    pattern = (
        r"RESIDENTIAL|RENTAL|APARTMENT|CONDOMINIUM|CONDO|DUPLEX|TRIPLEX|"
        r"ONE FAMILY|HOUSEHOLD UNIT|MOBILE HOME"
    )

    for col in desc_cols:
        if col in df.columns:
            mask = df[col].astype(str).str.upper().str.contains(pattern, na=False, regex=True)
            if mask.any():
                return df[mask].copy()

    return df


def build_xy(df: pd.DataFrame):
    """Build feature matrix and target."""
    feature_candidates = [
        "YEARBUILT",
        "SIZEAREA",
        "FULLBATH",
        "THREEQUARTERBATH",
        "HALFBATH",
        "CALC_ACRES",
    ]
    feature_names = [c for c in feature_candidates if c in df.columns]

    if not feature_names:
        raise ValueError("No verified feature columns found in dataset.")

    X = df[feature_names].copy()
    y = df["APPRAISED_VALUE"].copy()

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    # remove impossible years if present
    if "YEARBUILT" in X.columns:
        X.loc[(X["YEARBUILT"] < 1800) | (X["YEARBUILT"] > 2100), "YEARBUILT"] = pd.NA

    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    X, y = X.align(y, join="inner", axis=0)

    if len(X) < 50:
        raise ValueError(f"Not enough usable rows for training: {len(X)}")

    return X.astype(float), y.astype(float), list(X.columns)


def train_model(X: pd.DataFrame, y: pd.Series):
    """Train linear regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2


# -------------------------------
# Load data
# -------------------------------
try:
    with st.spinner("Downloading and preparing data..."):
        df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.success("Data loaded successfully.")
st.write("Total rows loaded:", len(df))

with st.expander("Show available columns"):
    st.write(df.columns.tolist())

# -------------------------------
# Clean data
# -------------------------------
try:
    df = clean_target(df)
except Exception as e:
    st.error(f"Failed to clean APPRAISED_VALUE: {e}")
    st.stop()

use_residential = st.checkbox("Filter to residential properties", value=True)

if use_residential:
    df_use = residential_filter(df)
    if len(df_use) < 100:
        st.warning(
            f"Residential filter returned only {len(df_use)} rows. "
            "Using the full cleaned dataset instead."
        )
        df_use = df.copy()
else:
    df_use = df.copy()

st.write("Rows after target cleaning:", len(df))
st.write("Rows used for modeling:", len(df_use))

# Helpful diagnostics during development
with st.expander("Filter diagnostics"):
    for col in ["PROP_TYPE_CODE", "PROP_TYPE_CODE_DESC", "LAND_USE_CODE", "LAND_USE_CODE_DESC"]:
        if col in df.columns:
            sample_vals = df[col].dropna().astype(str).head(10).tolist()
            st.write(f"{col} sample values:", sample_vals)

if len(df_use) == 0:
    st.error("No rows available after filtering.")
    st.stop()

# -------------------------------
# Train model
# -------------------------------
try:
    X, y, feature_names = build_xy(df_use)
    model, mae, r2 = train_model(X, y)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

st.success("Model trained successfully.")
col1, col2 = st.columns(2)
col1.metric("MAE", f"${mae:,.0f}")
col2.metric("R²", f"{r2:.3f}")

st.subheader("Features used")
st.write(feature_names)

# -------------------------------
# Prediction UI
# -------------------------------
st.subheader("Predict APPRAISED_VALUE")
st.caption("Enter property characteristics to estimate appraised value.")

user_input = {}
ui_cols = st.columns(2)

for i, col in enumerate(feature_names):
    default_val = float(pd.to_numeric(df_use[col], errors="coerce").median())
    with ui_cols[i % 2]:
        user_input[col] = st.number_input(col, value=default_val)

X_user = pd.DataFrame([user_input])[feature_names].astype(float)
prediction = float(model.predict(X_user)[0])

st.write("### Predicted APPRAISED_VALUE")
st.write(f"**${prediction:,.0f}**")
