import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess_prediction_data(input_file, output_file, scaler_file, encoder_file):
    try:
        # Load new property data
        data = pd.read_csv(input_file)
        print("Loaded new data for prediction.")
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        return

    # Load saved encoders and scaler
    try:
        scaler = joblib.load(scaler_file)
        encoders = joblib.load(encoder_file)
        print("Loaded scaler and encoders.")
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        return

    # Encode categorical features
    for col in ["Neighborhood", "Property Type"]:
        if col in encoders:
            encoder = encoders[col]
            data[col] = data[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        else:
            raise ValueError(f"No encoder found for column: {col}")
    print("Encoded categorical features using saved encoders.")

    # Validate input data for required columns
    required_columns = [
        "Year Built", "Square Footage", "Bedrooms", "Bathrooms", "Purchase Price",
        "Down Payment", "Loan Amount", "Property Taxes", "Insurance Costs",
        "HOA Fees", "Maintenance Costs", "Renovation Costs", "Current Market Rent",
        "Vacancy Rate", "Interest Rate (%)"
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in input data: {missing_columns}")

    # Generate derived features
    print("Generating derived features...")
    data["Property Age"] = 2025 - data["Year Built"]
    data["Net Operating Income (NOI)"] = (data["Current Market Rent"] * 12 * (1 - data["Vacancy Rate"])) - (
        data["Property Taxes"] + data["Insurance Costs"] + data["Maintenance Costs"] + data["HOA Fees"] * 12
    )
    data["Rental Yield"] = data["Current Market Rent"] * 12 / data["Purchase Price"].replace(0, np.nan)
    data["Debt to Income Ratio"] = data["Loan Amount"] / data["Net Operating Income (NOI)"].replace(0, np.nan)
    data["Annual Cash Flow"] = (
        data["Net Operating Income (NOI)"] - 
        (data["Loan Amount"] * (data["Interest Rate (%)"] / 100))
    )
    data["Cap Rate (%)"] = (data["Net Operating Income (NOI)"] / data["Purchase Price"]) * 100
    data["Total ROI (%)"] = ((data["Annual Cash Flow"] / data["Down Payment"]) * 100).replace(np.nan, 0)
    print("Derived features added.")

    # Scale numerical features
    numeric_features = [
        "Year Built", "Square Footage", "Bedrooms", "Bathrooms", "Purchase Price",
        "Down Payment", "Loan Amount", "Property Taxes", "Insurance Costs",
        "HOA Fees", "Maintenance Costs", "Renovation Costs", "Current Market Rent",
        "Net Operating Income (NOI)", "Rental Yield", "Debt to Income Ratio",
        "Annual Cash Flow", "Cap Rate (%)", "Total ROI (%)"
    ]

    # Ensure only existing numeric features in the input data are scaled
    numeric_features = [col for col in numeric_features if col in data.columns]

    try:
        data[numeric_features] = scaler.transform(data[numeric_features])
        print("Scaled numerical features using saved scaler.")
    except ValueError as e:
        print(f"Error during scaling: {e}")
        return

    # Save processed data
    data.to_csv(output_file, index=False)
    print(f"Processed prediction data saved to {output_file}.")

# Example usage
if __name__ == "__main__":
    preprocess_prediction_data(
        input_file="../data/new_property_data.csv",         # Input file for new data
        output_file="../data/processed_new_property_data.csv",  # Output file for processed data
        scaler_file="../models/scaler.pkl",                # Scaler file from training
        encoder_file="../models/encoders.pkl"              # Encoders file from training
    )
