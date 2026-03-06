import io
import zipfile
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hamilton County Home Value Predictor", layout="wide")

DATA_URL = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"


# -----------------------------
# Download and load data
# -----------------------------
@st.cache_data
def load_data():

    response = requests.get(DATA_URL)
    z = zipfile.ZipFile(io.BytesIO(response.content))

    csv_name = [f for f in z.namelist() if f.endswith(".csv")][0]

    df = pd.read_csv(z.open(csv_name), low_memory=False)

    df.columns = [c.strip().upper() for c in df.columns]

    return df


# -----------------------------
# Convert column to numeric
# -----------------------------
def try_numeric(series):

    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.replace("$", ""),
        errors="coerce"
    )


# -----------------------------
# Clean dataset
# -----------------------------
def clean_dataset(df):

    df = df.copy()

    df["APPRAISED_VALUE"] = try_numeric(df["APPRAISED_VALUE"])

    df = df[df["APPRAISED_VALUE"] > 1000]

    return df


# -----------------------------
# Find usable numeric columns
# -----------------------------
def detect_features(df):

    usable = []

    for col in df.columns:

        if col == "APPRAISED_VALUE":
            continue

        numeric = try_numeric(df[col])

        if numeric.notna().sum() > 2000 and numeric.nunique() > 10:
            usable.append(col)

    return usable


# -----------------------------
# Build model dataset
# -----------------------------
def build_xy(df, features):

    X = pd.DataFrame()

    for col in features:
        X[col] = try_numeric(df[col])

    y = df["APPRAISED_VALUE"]

    X = X.fillna(X.median())

    return X, y


# -----------------------------
# Train model
# -----------------------------
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return model, mae, r2


# -----------------------------
# App
# -----------------------------
st.title("Hamilton County Home Value Predictor")

df = load_data()

df = clean_dataset(df)

features = detect_features(df)

st.sidebar.header("Select model features")

selected_features = st.sidebar.multiselect(
    "Features",
    features,
    default=features[:5]
)

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

X, y = build_xy(df, selected_features)

model, mae, r2 = train_model(X, y)

st.sidebar.header("Enter Property Features")

user_input = {}

for f in selected_features:

    default = float(X[f].median())

    user_input[f] = st.sidebar.number_input(f, value=default)

X_user = pd.DataFrame([user_input])

prediction = model.predict(X_user)[0]

st.subheader("Predicted Appraised Value")

st.metric("Estimated Value", f"${prediction:,.0f}")

col1, col2 = st.columns(2)

with col1:
    st.write("Inputs")
    st.write(user_input)

with col2:
    st.write("Model stats")
    st.write(f"MAE: ${mae:,.0f}")
    st.write(f"R²: {r2:.3f}")

# Optional debug
with st.expander("Detected numeric columns"):
    st.write(features)
