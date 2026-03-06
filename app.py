import streamlit as st
import pandas as pd
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Page Configuration ---
st.set_page_config(page_title="Hamilton County Properties Predictor", layout="centered")

# --- 1. Load and Clean Data via ZIP URL ---
@st.cache_data(show_spinner=False)
def load_data():
    # 👉 PASTE YOUR URL TO THE .ZIP FILE HERE 👈
    data_url = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip"
    
    try:
        # Download the ZIP file from the URL
        response = requests.get(data_url, timeout=120)
        response.raise_for_status()
        
        # Extract the CSV from the ZIP in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the first CSV file in the zip archive
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            
            if not csv_files:
                st.error("No CSV file found inside the downloaded ZIP.")
                st.stop()
            
            target_csv = csv_files[0] 
            with z.open(target_csv) as f:
                df = pd.read_csv(f, low_memory=False)
                
    except Exception as e:
        st.error(f"Failed to load or unzip dataset from URL. Error: {e}")
        st.stop()
    
    # Standardize column names
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    if "APPRAISED_VALUE" not in df.columns:
        st.error(f"The extracted file ({target_csv}) does not contain the target variable 'APPRAISED_VALUE'.")
        st.stop()

    # Clean target variable [cite: 11]
    df["APPRAISED_VALUE"] = pd.to_numeric(
        df["APPRAISED_VALUE"].astype(str).str.replace(r'[$,]', '', regex=True), errors="coerce"
    )
    df_model = df.dropna(subset=["APPRAISED_VALUE"])
    df_model = df_model[df_model["APPRAISED_VALUE"] > 0]
    
    # Filter to residential parcels if the column exists [cite: 11]
    if "LAND_USE_CODE_DESC" in df_model.columns:
        df_model = df_model[df_model["LAND_USE_CODE_DESC"].astype(str).str.contains("One Family", na=False)]
        
    return df_model

# --- 2. Train Model ---
@st.cache_resource(show_spinner=False)
def train_model(df_model):
    target = "APPRAISED_VALUE"
    
    # Recommended features from the lab [cite: 12]
    expected_num = ["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]
    expected_cat = ["ZONING_DESC", "NEIGHBORHOOD_CODE_DESC", "LAND_USE_CODE_DESC", "PROPERTY_TYPE_CODE_DESC"]

    # Only use features that actually exist in the downloaded CSV
    num_features = [f for f in expected_num if f in df_model.columns]
    cat_features = [f for f in expected_cat if f in df_model.columns]

    if not num_features and not cat_features:
        st.error("None of the required predictor features were found in the dataset! Check your data source.")
        st.stop()

    X = df_model[num_features + cat_features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return model, mae, r2, df_model, num_features, cat_features

# --- Main App Execution ---

st.title("Hamilton County Properties Predictor")
st.write("Enter the property characteristics below to predict the appraised value using a machine learning model.")

with st.spinner("Downloading ZIP file and training model..."):
    df_model = load_data()
    model, mae, r2, df_model, num_features, cat_features = train_model(df_model)

# --- 3. User Inputs (Sidebar) ---
st.sidebar.header("Property Characteristics")
user_inputs = {}

# Dynamically generate inputs based on available numerical features
for feature in num_features:
    user_inputs[feature] = st.sidebar.number_input(f"{feature} (Numerical)", min_value=0.0, value=0.0, step=100.0)

# Dynamically generate dropdowns based on available categorical features
for feature in cat_features:
    options = df_model[feature].dropna().unique().tolist()
    user_inputs[feature] = st.sidebar.selectbox(f"{feature} (Categorical)", options=options)

# --- 4. Prediction ---
input_df = pd.DataFrame([user_inputs])

if st.sidebar.button("Predict Appraised Value"):
    prediction = model.predict(input_df)[0]
    
    st.subheader("Results")
    st.metric(label="Predicted Appraised Value", value=f"${prediction:,.2f}") 
    
    st.divider()
    st.write("### Model Performance Metrics")
    st.write(f"- **Mean Absolute Error (MAE):** ${mae:,.2f}") # Output requested by rubric [cite: 14]
    st.write(f"- **R² Score:** {r2:.3f}") # Output requested by rubric [cite: 14]

st.caption("This is for educational demonstration only.") # Mandatory disclaimer [cite: 16]
