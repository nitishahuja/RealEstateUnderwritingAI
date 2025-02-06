import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

def preprocess_training_data(input_file, output_file, scaler_file, encoder_file, feature_file):
    """
    Preprocesses training data for real estate investment predictions:
    - Encodes categorical variables (State, Neighborhood, Property Type)
    - Computes derived financial metrics (NOI, ROI, Cap Rate, etc.)
    - Applies log transformations & scaling to numerical features
    - Ensures target columns exist for training
    - Saves processed dataset & preprocessing artifacts (scaler, encoder, feature names)
    """
    
    print("📥 Loading raw training data...")
    data = pd.read_csv(input_file)
    print("✅ Loaded training data.")

    # **1️⃣ Handle Missing Values in Categorical Features**
    categorical_features = ["State", "Neighborhood", "Property Type"]
    data[categorical_features] = data[categorical_features].fillna("Unknown")
    
    # **2️⃣ One-Hot Encode Categorical Features**
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded_features = encoder.fit_transform(data[categorical_features])
    
    # Convert to DataFrame with proper column names
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    
    # Drop original categorical columns and concatenate encoded ones
    data = data.drop(columns=categorical_features).reset_index(drop=True)
    data = pd.concat([data, encoded_df], axis=1)
    
    print("✅ Encoded categorical features.")

    # **3️⃣ Derived Feature Calculations**
    current_year = 2025
    data["Property Age"] = current_year - data["Year Built"]
    
    data["Net Operating Income (NOI)"] = (
        (data["Current Market Rent"] * 12 * (1 - data["Vacancy Rate"])) 
        - (data["Property Taxes"] + data["Insurance Costs"] + data["Maintenance Costs"] + (data["HOA Fees"] * 12))
    )

    # Handle division errors
    data["Rental Yield"] = np.where(
        data["Purchase Price"] > 0,
        (data["Current Market Rent"] * 12 / data["Purchase Price"]),
        0
    )
    
    data["Debt to Income Ratio"] = np.where(
        data["Net Operating Income (NOI)"] > 0,
        data["Loan Amount"] / data["Net Operating Income (NOI)"],
        0
    )

    data["Annual Cash Flow"] = data["Net Operating Income (NOI)"] - (data["Loan Amount"] * ((data["Interest Rate (%)"] / 100) / 12) * 12)

    data["Cap Rate (%)"] = np.where(
        data["Purchase Price"] > 0,
        (data["Net Operating Income (NOI)"] / data["Purchase Price"]) * 100,
        0
    )

    data["Total ROI (%)"] = np.where(
        (data["Down Payment"] + data["Renovation Costs"]) > 0,
        ((data["Annual Cash Flow"] / (data["Down Payment"] + data["Renovation Costs"])) * 100),
        0
    )

    print("✅ Derived features calculated.")

    # **4️⃣ Ensure Target Columns Exist Before Training**
    target_columns = ["Total ROI (%)", "Net Operating Income (NOI)", "Cap Rate (%)", "Annual Cash Flow"]
    missing_targets = [col for col in target_columns if col not in data.columns]

    if missing_targets:
        raise ValueError(f"🚨 ERROR: Missing target columns: {missing_targets}")

    print("✅ Verified target columns exist.")

    # **5️⃣ Prevent Negative Values in Financial Metrics**
    financial_columns = ["Net Operating Income (NOI)", "Annual Cash Flow", "Cap Rate (%)", "Total ROI (%)"]
    for col in financial_columns:
        data[col] = data[col].clip(lower=0)

    # **6️⃣ Apply Log Transform for Skewed Features**
    skewed_features = ["Square Footage", "Purchase Price", "Loan Amount", "Property Taxes", "Current Market Rent"]
    for col in skewed_features:
        data[col] = np.log1p(data[col])  # log(1 + x) prevents log(0)

    print("✅ Applied log transformations for skewed features.")

    # **7️⃣ Scaling Features Using MinMaxScaler**
    numeric_features = [
        "Year Built", "Property Age", "Square Footage", "Bedrooms", "Bathrooms", "Purchase Price",
        "Down Payment", "Loan Amount", "Property Taxes", "Insurance Costs", "HOA Fees",
        "Maintenance Costs", "Renovation Costs", "Current Market Rent",
        "Net Operating Income (NOI)", "Rental Yield", "Debt to Income Ratio",
        "Annual Cash Flow", "Cap Rate (%)", "Total ROI (%)"
    ]

    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    print("✅ Scaled numerical features.")

    # **8️⃣ Save Feature Names & Preprocessing Objects**
    feature_names = list(encoded_df.columns) + numeric_features
    joblib.dump(feature_names, feature_file)
    joblib.dump(encoder, encoder_file)
    joblib.dump(scaler, scaler_file)

    print("✅ Saved encoder, scaler, and feature names.")

    # **9️⃣ Save Processed Data**
    data.to_csv(output_file, index=False)
    print(f"✅ Processed training data saved to: {output_file}")

# **Example Usage**
if __name__ == "__main__":
    preprocess_training_data(
        input_file="../data/raw_training_data.csv",
        output_file="../data/processed_training_data.csv",
        scaler_file="../models/scaler.pkl",
        encoder_file="../models/encoder.pkl",
        feature_file="../models/feature_names.pkl"
    )
