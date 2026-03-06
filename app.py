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


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    parcel_zip = download_zip_bytes(PARCEL_ZIP_URL)
    return read_parcel_csv_from_zip(parcel_zip)


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "APPRAISED_VALUE" not in df.columns:
        raise KeyError("APPRAISED_VALUE column not found.")

    df["APPRAISED_VALUE"] = clean_numeric_series(df["APPRAISED_VALUE"])
    df = df.dropna(subset=["APPRAISED_VALUE"])
    df = df[df["APPRAISED_VALUE"] > 1000].copy()
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


def auto_detect_numeric_features(df: pd.DataFrame) -> list[str]:
    """
    Automatically find usable numeric parcel columns.
    """
    exclude_cols = {
        "APPRAISED_VALUE",
        "MAP", "GROUP", "PARCEL",
        "OWNER_NAME", "OWNER_ADDRESS",
        "PROPERTY_ADDRESS", "CITY", "STATE", "ZIP",
        "PROP_TYPE_CODE_DESC", "LAND_USE_CODE_DESC", "CURRENT_USE_CODE_DESC",
        "TAX_DISTRICT", "DEED_BOOK", "DEED_PAGE",
    }

    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    usable = []
    for col in candidate_cols:
        s_num = clean_numeric_series(df[col])

        non_missing = s_num.notna().sum()
        nunique = s_num.nunique(dropna=True)

        if non_missing >= 1000 and nunique >= 5:
            usable.append(col)

    return usable


def choose_default_features(df: pd.DataFrame, max_features: int = 8) -> list[str]:
    """
    Prefer interpretable housing-related names first, then fall back to auto-detected numeric columns.
    """
    preferred = [
        "CALC_ACRES",
        "LAND_SQUARE_FOOTAGE",
        "TOTAL_ROOMS",
        "BEDROOMS",
        "YEAR_BUILT",
        "STORIES",
        "GRADE",
        "CONDITION",
        "BUILDING_SQUARE_FEET",
        "LIVING_AREA",
        "HEATED_AREA",
        "BATHS",
        "FULL_BATHS",
        "HALF_BATHS",
    ]

    detected = auto_detect_numeric_features(df)

    selected = [c for c in preferred if c in detected]

    for col in detected:
        if col not in selected:
            selected.append(col)

    return selected[:max_features]


def build_xy(df: pd.DataFrame, selected_features: list[str]):
    if not selected_features:
        raise ValueError("No usable feature columns were selected.")

    X = df[selected_features].copy()
    y = df["APPRAISED_VALUE"].copy()

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    # Light sanity cleaning for any year-like columns
    for col in X.columns:
        if "YEAR" in col:
            X.loc[(X[col] < 1800) | (X[col] > 2100), col] = pd.NA

    # Remove columns that collapse after cleaning
    usable_cols = [c for c in X.columns if X[c].notna().sum() >= 1000 and X[c].nunique(dropna=True) >= 5]
    X = X[usable_cols].copy()

    if X.empty:
        raise ValueError("No usable predictor columns remain after cleaning.")

    feature_defaults = X.median(numeric_only=True).to_dict()
    X = X.fillna(pd.Series(feature_defaults))
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(pd.Series(feature_defaults))

    X, y = X.align(y, join="inner", axis=0)

    if len(X) < 1000:
        raise ValueError(f"Not enough usable rows for training: {len(X)}")

    return X.astype(float), y.astype(float), list(X.columns), feature_defaults


def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2


# -----------------------------
# Load and clean first
# -----------------------------
st.title("Hamilton County Home Value Predictor")
st.write("Enter property features to estimate appraised value.")

show_debug = st.sidebar.checkbox("Show debug info", value=True)
use_residential = st.sidebar.checkbox("Use residential filter", value=True)

try:
    with st.spinner("Loading parcel data..."):
        df = load_data()
        df = clean_target(df)

        if use_residential:
            df_filtered = residential_filter(df)
            if len(df_filtered) >= 1000:
                df = df_filtered
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

# -----------------------------
# Detect features before train
# -----------------------------
detected_features = auto_detect_numeric_features(df)
default_features = choose_default_features(df, max_features=8)

if show_debug:
    with st.expander("Available columns", expanded=False):
        st.write(df.columns.tolist())

    with st.expander("Auto-detected numeric feature candidates", expanded=True):
        st.write(detected_features)

    with st.expander("Preview of key data", expanded=False):
        preview_cols = ["APPRAISED_VALUE"] + default_features[:6]
        preview_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(df[preview_cols].head(10))

# Let user choose which detected numeric fields to use
selected_features = st.sidebar.multiselect(
    "Features used by model",
    options=detected_features,
    default=default_features,
)

if not selected_features:
    st.warning("Please select at least one feature in the sidebar.")
    st.stop()

# -----------------------------
# Build and train
# -----------------------------
try:
    with st.spinner("Training model..."):
        X, y, feature_names, feature_defaults = build_xy(df, selected_features)
        model, mae, r2 = train_model(X, y)
except Exception as e:
    st.error(f"Model setup failed: {e}")
    st.stop()

# -----------------------------
# Sidebar input controls
# -----------------------------
st.sidebar.header("Property Inputs")

label_map = {
    "CALC_ACRES": "Lot Size (acres)",
    "LAND_SQUARE_FOOTAGE": "Land Area (sq ft)",
    "TOTAL_ROOMS": "Total Rooms",
    "BEDROOMS": "Bedrooms",
    "YEAR_BUILT": "Year Built",
    "STORIES": "Stories",
    "GRADE": "Grade",
    "CONDITION": "Condition",
    "BUILDING_SQUARE_FEET": "Building Area (sq ft)",
    "LIVING_AREA": "Living Area (sq ft)",
    "HEATED_AREA": "Heated Area (sq ft)",
}

user_input = {}
for col in feature_names:
    default_val = feature_defaults.get(col, 0.0)
    if pd.isna(default_val):
        default_val = 0.0

    label = label_map.get(col, col)

    is_integer_like = (
        ("YEAR" in col) or
        ("ROOM" in col) or
        ("BED" in col) or
        ("BATH" in col) or
        ("STORY" in col) or
        ("GRADE" in col) or
        ("CONDITION" in col)
    )

    if is_integer_like:
        user_input[col] = st.sidebar.number_input(
            label,
            min_value=0,
            value=int(round(float(default_val))),
            step=1,
        )
    else:
        user_input[col] = st.sidebar.number_input(
            label,
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

# -----------------------------
# Main result display
# -----------------------------
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
    "This tool is for educational demonstration only. Predictions are based on a simple linear regression model trained on public assessor data."
)

if show_debug:
    st.divider()
    st.subheader("Debug Information")

    with st.expander("Non-missing counts for selected features", expanded=False):
        st.write(df[feature_names].notna().sum())

    with st.expander("Cleaned feature matrix preview", expanded=False):
        st.dataframe(X.head(10))

    with st.expander("Target summary", expanded=False):
        st.write(y.describe())

    with st.expander("Prediction row sent to model", expanded=False):
        st.dataframe(X_user)
