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

# --- 1. Load and Clean Data via URL ---
@st.cache_data
def load_data():
    # 👉 PASTE YOUR RAW URL HERE 👈
    # Example GitHub Raw URL or Dropbox direct link (dl=1)
    data_url = "https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip" 
    
    try:
        # If using your compressed CSV:
        df = pd.read_csv(data_url, compression="gzip")
        
        # If you are using the original Excel file via Dropbox/Drive, comment out the line above 
        # and uncomment the two lines below:
        # use_cols = ["APPRAISED_VALUE", "LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES", "ZONING_DESC", "NEIGHBORHOOD_CODE_DESC", "LAND_USE_CODE_DESC", "PROPERTY_TYPE_CODE_DESC"]
        # df = pd.read_excel(data_url, engine="openpyxl", usecols=use_cols)

    except Exception as e:
        st.error(f"Failed to load dataset from URL. Error: {e}")
        st.stop()
    
    # Cleaning steps required by the lab
    df_model = df.dropna(subset=["APPRAISED_VALUE"]) # Drop missing targets
    df_model = df_model[df_model["APPRAISED_VALUE"] > 0] # Remove invalid values
    
    # Filter to residential parcels
    df_model = df_model[df_model["LAND_USE_CODE_DESC"].astype(str).str.contains("One Family", na=False)]
    
    return df_model

# --- 2. Train Model ---
@st.cache_resource
def train_model(df_model):
    target = "APPRAISED_VALUE"
    
    num_features = ["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]
    cat_features = ["ZONING_DESC", "NEIGHBORHOOD_CODE_DESC", "LAND_USE_CODE_DESC", "PROPERTY_TYPE_CODE_DESC"]

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
    
    return model, mae, r2, df_model

# --- Main App Execution ---

st.title("Hamilton County Properties Predictor")
st.write("Enter the property characteristics below to predict the appraised value using a machine learning model.")

with st.spinner("Downloading data from URL and training model..."):
    df_model = load_data()
    model, mae, r2, df_model = train_model(df_model)

# --- 3. User Inputs (Sidebar) ---
st.sidebar.header("Property Characteristics")

land_value = st.sidebar.number_input("Land Value ($)", min_value=0, value=50000, step=1000)
build_value = st.sidebar.number_input("Build Value ($)", min_value=0, value=150000, step=1000)
yarditems_value = st.sidebar.number_input("Yard Items Value ($)", min_value=0, value=0, step=500)
calc_acres = st.sidebar.number_input("Calculated Acres", min_value=0.0, value=0.25, step=0.1)

zoning_desc = st.sidebar.selectbox("Zoning Description", options=df_model["ZONING_DESC"].dropna().unique())
neighborhood = st.sidebar.selectbox("Neighborhood Code", options=df_model["NEIGHBORHOOD_CODE_DESC"].dropna().unique())
land_use = st.sidebar.selectbox("Land Use", options=df_model["LAND_USE_CODE_DESC"].dropna().unique())
prop_type = st.sidebar.selectbox("Property Type", options=df_model["PROPERTY_TYPE_CODE_DESC"].dropna().unique())

# --- 4. Prediction ---
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
    st.metric(label="Predicted Appraised Value", value=f"${prediction:,.2f}") 
    
    st.divider()
    st.write("### Model Performance Metrics")
    st.write(f"- **Mean Absolute Error (MAE):** ${mae:,.2f}")
    st.write(f"- **R² Score:** {r2:.3f}")

st.caption("This is for educational demonstration only.")
