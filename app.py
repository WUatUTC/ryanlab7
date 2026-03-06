import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Page Configuration ---
st.set_page_config(page_title="Hamilton County Properties Predictor", layout="centered")

# --- 1. Load and Clean Data ---
@st.cache_data
def load_data():
    # Load the compressed CSV to bypass GitHub's file size limits
    try:
        df = pd.read_csv("clean_housing_data.csv.gz", compression="gzip")
    except FileNotFoundError:
        st.error("Dataset not found! Ensure 'clean_housing_data.csv.gz' is uploaded to your GitHub repository.")
        st.stop()
    
    # Cleaning steps required by the lab
    df_model = df.dropna(subset=["APPRAISED_VALUE"]) # Ensure no missing targets [cite: 11]
    df_model = df_model[df_model["APPRAISED_VALUE"] > 0] # Remove invalid values [cite: 11]
    
    # Filter to residential parcels [cite: 11]
    df_model = df_model[df_model["LAND_USE_CODE_DESC"].astype(str).str.contains("One Family", na=False)]
    
    return df_model

# --- 2. Train Model ---
@st.cache_resource
def train_model(df_model):
    target = "APPRAISED_VALUE" [cite: 4]
    
    # Recommended starter predictors [cite: 12]
    num_features = ["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]
    cat_features = ["ZONING_DESC", "NEIGHBORHOOD_CODE_DESC", "LAND_USE_CODE_DESC", "PROPERTY_TYPE_CODE_DESC"]

    # Ensure all required columns exist in the dataframe
    missing_cols = [col for col in num_features + cat_features if col not in df_model.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}. Did you include them when shrinking the file?")
        st.stop()

    X = df_model[num_features + cat_features]
    y = df_model[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) [cite: 12]

    # Preprocessing pipelines [cite: 12, 13]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    # Combine preprocessing and model into a single pipeline 
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("regressor", LinearRegression())
    ])

    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate 
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return model, mae, r2, df_model

# --- Main App Execution ---

# App title + short description [cite: 15]
st.title("Hamilton County Properties Predictor")
st.write("Enter the property characteristics below to predict the appraised value using a machine learning model.")

with st.spinner("Loading data and training model..."):
    df_model = load_data()
    model, mae, r2, df_model = train_model(df_model)

# --- 3. User Inputs (Sidebar) ---
st.sidebar.header("Property Characteristics")

# Numerical inputs [cite: 15]
land_value = st.sidebar.number_input("Land Value ($)", min_value=0, value=50000, step=1000)
build_value = st.sidebar.number_input("Build Value ($)", min_value=0, value=150000, step=1000)
yarditems_value = st.sidebar.number_input("Yard Items Value ($)", min_value=0, value=0, step=500)
calc_acres = st.sidebar.number_input("Calculated Acres", min_value=0.0, value=0.25, step=0.1)

# Categorical inputs [cite: 15]
zoning_desc = st.sidebar.selectbox("Zoning Description", options=df_model["ZONING_DESC"].dropna().unique())
neighborhood = st.sidebar.selectbox("Neighborhood Code", options=df_model["NEIGHBORHOOD_CODE_DESC"].dropna().unique())
land_use = st.sidebar.selectbox("Land Use", options=df_model["LAND_USE_CODE_DESC"].dropna().unique())
prop_type = st.sidebar.selectbox("Property Type", options=df_model["PROPERTY_TYPE_CODE_DESC"].dropna().unique())

# --- 4. Prediction ---
# Create a dataframe for the user's input
input_data = pd.DataFrame({
    "LAND_VALUE": [land_value],
    "BUILD_VALUE": [build_value],
    "YARDITEMS_VALUE": [yarditems_value],
    "CALC_ACRES": [calc_acres],
    "ZONING_DESC": [zoning_desc],
    "NEIGHBORHOOD_CODE_DESC": [neighborhood],
    "LAND_USE_CODE_DESC": [land_use],
    "PROPERTY_TYPE_CODE_DESC": [prop_type]
})

if st.sidebar.button("Predict Appraised Value"):
    prediction = model.predict(input_data)[0]
    
    st.subheader("Results")
    # Output predicted appraised value [cite: 15, 16]
    st.metric(label="Predicted Appraised Value", value=f"${prediction:,.2f}") 
    
    st.divider()
    st.write("### Model Performance Metrics")
    st.write(f"- **Mean Absolute Error (MAE):** ${mae:,.2f}")
    st.write(f"- **R² Score:** {r2:.3f}")

# Brief disclaimer [cite: 16]
st.caption("This is for educational demonstration only.")
