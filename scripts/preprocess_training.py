import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess_training_data(input_file, output_file, scaler_file, encoder_file):
    # Load training data
    data = pd.read_csv(input_file)
    print("Loaded training data.")

    # Encode categorical features
    encoders = {}
    for col in ["Neighborhood", "Property Type"]:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder
    print("Encoded categorical features.")

    # Generate derived features
    current_year = 2025
    data["Property Age"] = current_year - data["Year Built"]
    data["Net Operating Income (NOI)"] = (
        (data["Current Market Rent"] * 12 * (1 - data["Vacancy Rate"])) 
        - (data["Property Taxes"] + data["Insurance Costs"] + data["Maintenance Costs"] + (data["HOA Fees"] * 12))
    )
    data["Rental Yield"] = (data["Current Market Rent"] * 12 / data["Purchase Price"].replace(0, np.nan)).replace(np.inf, 0)
    data["Debt to Income Ratio"] = (
        data["Loan Amount"] / data["Net Operating Income (NOI)"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0)
    data["Annual Cash Flow"] = (
        data["Net Operating Income (NOI)"] - 
        (data["Loan Amount"] * ((data["Interest Rate (%)"] / 100) / 12) * 12)
    )
    data["Cap Rate (%)"] = (
        (data["Net Operating Income (NOI)"] / data["Purchase Price"].replace(0, np.nan)) * 100
    ).replace(np.inf, 0)
    data["Total ROI (%)"] = (
        ((data["Annual Cash Flow"] / (data["Down Payment"] + data["Renovation Costs"])) * 100)
    ).replace([np.inf, -np.inf], 0).fillna(0)
    print("Derived features added.")

    # Ensure no negative or unrealistic values in critical metrics
    derived_columns = ["Net Operating Income (NOI)", "Annual Cash Flow", "Cap Rate (%)", "Total ROI (%)"]
    for col in derived_columns:
        data[col] = data[col].clip(lower=0)

    # Scale numerical features
    numeric_features = [
        "Year Built", "Square Footage", "Bedrooms", "Bathrooms", "Purchase Price",
        "Down Payment", "Loan Amount", "Property Taxes", "Insurance Costs",
        "HOA Fees", "Maintenance Costs", "Renovation Costs", "Current Market Rent",
        "Net Operating Income (NOI)", "Rental Yield", "Debt to Income Ratio",
        "Annual Cash Flow", "Cap Rate (%)", "Total ROI (%)"
    ]
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    print("Scaled numerical features.")

    # Save processed data and transformations
    data.to_csv(output_file, index=False)
    joblib.dump(scaler, scaler_file)
    joblib.dump(encoders, encoder_file)
    print(f"Processed training data saved to {output_file}.")
    print(f"Scaler and encoders saved.")

# Example usage
if __name__ == "__main__":
    preprocess_training_data(
        input_file="../data/raw_training_data.csv",
        output_file="../data/processed_training_data.csv",
        scaler_file="../models/scaler.pkl",
        encoder_file="../models/encoders.pkl"
    )
