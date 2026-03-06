import io
import zipfile

import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Hamilton County Home Value Predictor", layout="wide")

PARCEL_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"
BUILDING_ZIP_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorBuildingExport.zip"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def download_zip_bytes(url: str) -> bytes:
    """Download ZIP file from URL and return raw bytes."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names uppercase and trimmed."""
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Convert mixed text/numeric values to numeric."""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
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
    Try CSV first. If TXT exists, parse as fixed-width text.
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
                # Current tentative fixed-width locations used for key building fields
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
    """Download parcel and building data and merge them."""
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

    agg_map = {c: "max" for c in keep_cols if c not in merge_cols}
    if agg_map:
        df_building = df_building.groupby(merge_cols, as_index=False).agg(agg_map)
    else:
        df_building = df_building.drop_duplicates(subset=merge_cols)

    df = pd.merge(df_parcel, df_building, on=merge_cols, how="left")
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Clean APPRAISED_VALUE and keep only positive values."""
    df = df.copy()
    if "APPRAISED_VALUE" not in df.columns:
        raise KeyError("APPRAISED_VALUE column not found.")

    df["APPRAISED_VALUE"] = clean_numeric_series(df["APPRAISED_VALUE"])
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 0].copy()
    return df


def residential_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Try to keep residential-like properties using coded fields first."""
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

    desc_cols = ["PROP_TYPE_CODE_DESC", "LAND_USE_CODE_DESC", "CURRENT_USE_CODE_DESC"]
    pattern = (
        r"RESIDENTIAL|RENTAL|APARTMENT|CONDOMINIUM|CONDO|DUPLEX|TRIPLEX|"
        r"ONE FAMILY|HOUSEHOLD UNIT|MOBILE HOME"
    )

    for col in desc_cols:
        if col in df.columns:
            mask = df[col].astype(str).str.upper().str.contains(pattern, na=False, regex=True)
            if mask.sum() >= 100:
                return df[mask].copy()

    return df


def build_xy(df: pd.DataFrame):
    """Build cleaned feature matrix and target."""
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
        raise ValueError("No expected feature columns were found.")

    X = df[feature_names].copy()
    y = df["APPRAISED_VALUE"].copy()

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    if "YEARBUILT" in X.columns:
        X.loc[(X["YEARBUILT"] < 1800) | (X["YEARBUILT"] > 2100), "YEARBUILT"] = pd.NA

    # Keep columns with a reasonable amount of usable data
    usable_cols = [c for c in X.columns if X[c].notna().sum() >= 100]
    X = X[usable_cols].copy()

    if X.empty:
        raise ValueError("No usable predictor columns remain after cleaning.")

    feature_defaults = X.median(numeric_only=True).to_dict()
    X = X.fillna(pd.Series(feature_defaults))
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(pd.Series(feature_defaults))

    X, y = X.align(y, join="inner", axis=0)

    if len(X) < 100:
        raise ValueError(f"Not enough usable rows for training: {len(X)}")

    return X.astype(float), y.astype(float), list(X.columns), feature_defaults


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


# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("Hamilton County Home Value Predictor")
st.write("Enter a few property features to estimate appraised value.")

show_debug = st.sidebar.checkbox("Show debug info", value=False)
use_residential = st.sidebar.checkbox("Use residential filter", value=True)

# --------------------------------------------------
# Load, clean, prepare, train
# --------------------------------------------------
try:
    with st.spinner("Loading data and training model..."):
        df = load_data()
        df = clean_target(df)

        if use_residential:
            df_filtered = residential_filter(df)
            if len(df_filtered) >= 100:
                df = df_filtered

        X, y, feature_names, feature_defaults = build_xy(df)
        model, mae, r2 = train_model(X, y)

except Exception as e:
    st.error(f"App setup failed: {e}")
    st.stop()

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("Property Features")

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

try:
    prediction = float(model.predict(X_user)[0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --------------------------------------------------
# Main result display
# --------------------------------------------------
st.subheader("Estimated Appraised Value")
st.metric("Predicted Value", f"${prediction:,.0f}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Inputs Used")
    for col in feature_names:
        st.write(f"**{label_map.get(col, col)}:** {user_input[col]}")

with col2:
    st.markdown("### Model Information")
    st.write(f"**Features used:** {', '.join(label_map.get(c, c) for c in feature_names)}")
    st.write(f"**Mean Absolute Error:** ${mae:,.0f}")
    st.write(f"**R²:** {r2:.3f}")

st.caption(
    "This app is for educational demonstration only. Predictions come from a simple linear regression model trained on public assessor data."
)

# --------------------------------------------------
# Optional debug section
# --------------------------------------------------
if show_debug:
    st.divider()
    st.subheader("Debug Information")

    debug_cols = [
        "APPRAISED_VALUE",
        "CALC_ACRES",
        "YEARBUILT",
        "SIZEAREA",
        "FULLBATH",
        "THREEQUARTERBATH",
        "HALFBATH",
    ]
    existing_debug_cols = [c for c in debug_cols if c in df.columns]

    with st.expander("First 10 rows of key columns", expanded=True):
        st.write("Available debug columns:", existing_debug_cols)
        st.dataframe(df[existing_debug_cols].head(10))

    with st.expander("Non-missing counts", expanded=True):
        st.write(df[existing_debug_cols].notna().sum())

    with st.expander("APPRAISED_VALUE summary", expanded=True):
        st.write(df["APPRAISED_VALUE"].describe())

    with st.expander("Cleaned feature matrix preview", expanded=True):
        st.write("Feature names used:", feature_names)
        st.write("Non-missing counts in X:")
        st.write(X.notna().sum())
        st.dataframe(X.head(10))

    with st.expander("Prediction row sent to model", expanded=True):
        st.dataframe(X_user)
