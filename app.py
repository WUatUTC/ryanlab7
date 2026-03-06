import io
import zipfile

import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Hamilton County Home Value Predictor", layout="wide")

PARCEL_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"
BUILDING_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorBuildingExport.zip"


def download_zip_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def clean_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def read_parcel_csv_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in parcel ZIP.")
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)
    return normalize_cols(df)


def read_building_file_from_zip(zip_bytes: bytes) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    parcel_zip = download_zip_bytes(PARCEL_ZIP_URL)
    building_zip = download_zip_bytes(BUILDING_ZIP_URL)

    df_parcel = read_parcel_csv_from_zip(parcel_zip)
    df_building = read_building_file_from_zip(building_zip)

    merge_cols = ["MAP", "GROUP", "PARCEL"]

    keep_cols = merge_cols + [
        "YEARBUILT",
        "SIZEAREA",
        "FULLBATH",
        "THREEQUARTERBATH",
        "HALFBATH",
    ]
    keep_cols = [c for c in keep_cols if c in df_building.columns]
    df_building = df_building[keep_cols].copy()

    agg_map = {c: "max" for c in keep_cols if c not in merge_cols}
    if agg_map:
        df_building = df_building.groupby(merge_cols, as_index=False).agg(agg_map)

    df = pd.merge(df_parcel, df_building, on=merge_cols, how="left")
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "APPRAISED_VALUE" not in df.columns:
        raise KeyError("APPRAISED_VALUE column not found.")

    df["APPRAISED_VALUE"] = clean_numeric_series(df["APPRAISED_VALUE"])
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0].copy()
    return df


def residential_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "PROP_TYPE_CODE" in df.columns:
        prop_code = pd.to_numeric(df["PROP_TYPE_CODE"], errors="coerce")
        mask = prop_code.isin([22, 32, 40])
        if mask.sum() >= 100:
            return df[mask].copy()

    if "LAND_USE_CODE" in df.columns:
        lu = pd.to_numeric(df["LAND_USE_CODE"], errors="coerce")
        mask = ((lu >= 100) & (lu < 200)) | lu.isin([111, 112, 113, 114, 115, 116, 117])
        if mask.sum() >= 100:
            return df[mask].copy()

    return df


def build_xy(df: pd.DataFrame):
    feature_candidates = [
        "YEARBUILT",
        "SIZEAREA",
        "FULLBATH",
        "THREEQUARTERBATH",
        "HALFBATH",
        "CALC_ACRES",
    ]
    feature_names = [c for c in feature_candidates if c in df.columns]

    X = df[feature_names].copy()
    y = df["APPRAISED_VALUE"].copy()

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    if "YEARBUILT" in X.columns:
        X.loc[(X["YEARBUILT"] < 1800) | (X["YEARBUILT"] > 2100), "YEARBUILT"] = pd.NA

    # Drop columns that are almost entirely missing
    usable_cols = [c for c in X.columns if X[c].notna().sum() >= 100]
    X = X[usable_cols].copy()

    if X.empty:
        raise ValueError("No usable predictor columns found after cleaning.")

    feature_defaults = X.median(numeric_only=True).to_dict()
    X = X.fillna(pd.Series(feature_defaults))

    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(pd.Series(feature_defaults))

    X, y = X.align(y, join="inner", axis=0)

    if len(X) < 100:
        raise ValueError(f"Not enough usable rows for training: {len(X)}")

    return X.astype(float), y.astype(float), list(X.columns), feature_defaults


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_data()
    df = clean_target(df)
    df = residential_filter(df)

    X, y, feature_names, feature_defaults = build_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2, feature_names, feature_defaults


# ----------------------------
# App UI
# ----------------------------
st.title("Hamilton County Home Value Predictor")
st.markdown("Enter a few property features in the sidebar to estimate appraised value.")

try:
    with st.spinner("Loading data and training model..."):
        model, mae, r2, feature_names, feature_defaults = train_model()
except Exception as e:
    st.error(f"App setup failed: {e}")
    st.stop()

# Sidebar inputs
st.sidebar.header("Enter Property Features")

label_map = {
    "YEARBUILT": "Year Built",
    "SIZEAREA": "Building Area (sq ft)",
    "FULLBATH": "Full Bathrooms",
    "THREEQUARTERBATH": "3/4 Bathrooms",
    "HALFBATH": "Half Bathrooms",
    "CALC_ACRES": "Lot Size (acres)",
}

user_input = {}
for col in feature_names:
    default_val = feature_defaults.get(col, 0.0)
    if pd.isna(default_val):
        default_val = 0.0

    nice_label = label_map.get(col, col)

    if col in ["YEARBUILT", "FULLBATH", "THREEQUARTERBATH", "HALFBATH"]:
        user_input[col] = st.sidebar.number_input(
            nice_label,
            min_value=0,
            value=int(round(float(default_val))),
            step=1,
        )
    else:
        user_input[col] = st.sidebar.number_input(
            nice_label,
            min_value=0.0,
            value=float(default_val),
            step=0.1,
        )

X_user = pd.DataFrame([user_input], columns=feature_names)
for col in X_user.columns:
    X_user[col] = pd.to_numeric(X_user[col], errors="coerce")

X_user = X_user.replace([float("inf"), float("-inf")], pd.NA)
X_user = X_user.fillna(pd.Series(feature_defaults))
X_user = X_user.astype(float)

prediction = float(model.predict(X_user)[0])

# Main display
st.subheader("Estimated Appraised Value")
st.metric(label="Predicted Value", value=f"${prediction:,.0f}")

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Property Inputs")
    for col in feature_names:
        st.write(f"**{label_map.get(col, col)}:** {user_input[col]}")

with c2:
    st.markdown("### Model Summary")
    st.write(f"**Features used:** {', '.join(label_map.get(c, c) for c in feature_names)}")
    st.write(f"**Mean Absolute Error:** ${mae:,.0f}")
    st.write(f"**R²:** {r2:.3f}")

st.caption(
    "This tool is for educational demonstration only. Predictions are based on a simple linear regression model trained on publicly available assessor data."
)
