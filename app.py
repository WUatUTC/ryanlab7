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
@st.cache_data(show_spinner=False)
def load_data():
    try:
        # Since the file is in your GitHub repo, Streamlit reads it directly!
        df = pd.read_csv("Housing_Hamilton_Compressed.csv.gz", compression="gzip")
    except FileNotFoundError:
        st.error("Dataset not found! Please make sure 'Housing_Hamilton_Compressed.csv.gz' is uploaded to your GitHub repository.")
        st.stop()
        
    # Apply lab cleaning steps
    # Drop rows with missing APPRAISED_VALUE [cite: 11]
    df_model = df.dropna(subset=["APPRAISED_VALUE"]) 
    
    # Remove clearly invalid values (e.g., APPRAISED_VALUE <= 0) [cite: 11]
    df_model = df_model[df_model["APPRAISED_VALUE"] > 0] 
    
    # Filter to residential parcels [cite: 11]
    if "LAND_USE_CODE_DESC" in df_model.columns:
        df_model = df_model[df_model["LAND_USE_CODE_DESC"].astype(str).str.contains("One Family", na=False)]
        
    # Safety net: Ensure the dataset isn't empty after cleaning
    if len(df_model) == 0:
        st.error("Error: The dataset has 0 rows after cleaning! Please check your data source.")
        st.stop()
        
    return df_model

# --- 2. Train Model ---
@st.cache_resource(show_spinner=False)
def train_model(df_model):
    target = "APPRAISED_VALUE"
    
    # Recommended starter predictors [cite: 12]
    num_features = ["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]
    cat_features = ["ZONING_DESC", "NEIGHBORHOOD_CODE_DESC", "LAND_USE_CODE_DESC", "PROPERTY_TYPE_CODE_DESC"]

    # Ensure all required columns exist
    missing_cols = [col for col in num_features + cat_features if col not in df_model.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        st.stop()

    X = df_model[num_features + cat_features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipelines [cite: 12]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")), # [cite: 13]
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    # Model training [cite: 14]
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
st.write("Enter the property characteristics below to predict the appraised value.")

with st.spinner("Loading compressed data and training model..."):
    df_model = load_data()
    model, mae, r2, df_model, num_features, cat_features = train_model(df_model)

# --- 3. User Inputs (Sidebar) ---
st.sidebar.header("Property Characteristics")
user_inputs = {}

# Minimum app requirements: inputs for LAND_VALUE, BUILD_VALUE, YARDITEMS_VALUE, CALC_ACRES [cite: 15]
for feature in num_features:
    # Set some sensible defaults based on typical values
    default_val = float(df_model[feature].median())
    user_inputs[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, value=default_val, step=100.0)

# Minimum app requirements: at least one categorical selector [cite: 15]
for feature in cat_features:
    options = df_model[feature].dropna().unique().tolist()
    user_inputs[feature] = st.sidebar.selectbox(f"{feature}", options=options)

# --- 4. Prediction ---
input_df = pd.DataFrame([user_inputs])

if st.sidebar.button("Predict Appraised Value"):
    prediction = model.predict(input_df)[0]
    
    st.subheader("Results")
    # Output: predicted appraised value [cite: 15, 16]
    st.metric(label="Predicted Appraised Value", value=f"${prediction:,.2f}") 
    
    st.divider()
    st.write("### Model Performance Metrics")
    # Report both MAE and R² [cite: 15]
    st.write(f"- **Mean Absolute Error (MAE):** ${mae:,.2f}") 
    st.write(f"- **R² Score:** {r2:.3f}") 

# Output: a brief disclaimer: “This is for educational demonstration only.” [cite: 16]
st.caption("This is for educational demonstration only.")
