import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_prediction_data(input_file, output_file, scaler_file, encoder_file, feature_file):
    """
    Preprocess new property data for predictions:
    - Encodes categorical variables using saved encoders
    - Applies standardization using the saved scaler
    - Ensures feature alignment with the trained model
    - Saves processed data ready for model predictions
    """

    # ✅ Step 1: Load the new dataset
    try:
        data = pd.read_csv(input_file)
        print("✅ Loaded new data for prediction.")
    except FileNotFoundError:
        print(f"🚨 Input file not found: {input_file}")
        return

    # ✅ Step 2: Load saved encoders, scaler, and feature names
    try:
        scaler = joblib.load(scaler_file)
        encoder = joblib.load(encoder_file)  # This should be OneHotEncoder
        feature_names = joblib.load(feature_file)
        print("✅ Loaded scaler, encoder, and feature names.")
    except FileNotFoundError as e:
        print(f"🚨 Missing file: {e}")
        return

    # ✅ Step 3: Encode categorical features using OneHotEncoder
    categorical_features = ["State", "Neighborhood", "Property Type"]

    if isinstance(encoder, OneHotEncoder):
        try:
            encoded_features = encoder.transform(data[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        except Exception as e:
            print(f"🚨 Error during encoding: {e}")
            return
        
        # Drop original categorical columns and concatenate encoded ones
        data = data.drop(columns=categorical_features).reset_index(drop=True)
        data = pd.concat([data, encoded_df], axis=1)
        print("✅ Encoded categorical features using saved OneHotEncoder.")
    else:
        print("🚨 Encoder file does not contain a valid OneHotEncoder.")

    # ✅ Step 4: Ensure feature alignment with trained model
    all_features = list(encoded_df.columns) + list(data.columns)
    final_features = [col for col in feature_names if col in all_features]

    missing_features = [col for col in feature_names if col not in data.columns]
    if missing_features:
        print(f"🚨 Warning: Some features are missing: {missing_features}. They will be filled with 0.")
        for col in missing_features:
            data[col] = 0  # Fill missing features with 0

    # ✅ Step 5: Scale numerical features using saved scaler
    try:
        data[final_features] = scaler.transform(data[final_features])
        print("✅ Scaled numerical features using saved scaler.")
    except ValueError as e:
        print(f"🚨 Error during scaling: {e}")
        return

    # ✅ Step 6: Save processed data
    data.to_csv(output_file, index=False)
    print(f"✅ Processed prediction data saved to {output_file}.")

# **Example Usage**
if __name__ == "__main__":
    preprocess_prediction_data(
        input_file="../data/sample_prediction_input.csv",  # New raw data for prediction
        output_file="../data/processed_prediction_data.csv",      # Output processed data
        scaler_file="../models/scaler.pkl",                       # Scaler from training
        encoder_file="../models/encoder.pkl",                     # OneHotEncoder from training
        feature_file="../models/feature_names.pkl"                # Feature names from training
    )
