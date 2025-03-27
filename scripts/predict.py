import pandas as pd
import joblib
import os
import numpy as np

def make_predictions(data_file, models_folder, scaler_file, output_file):
    """
    Loads trained models and makes predictions on new property data.
    
    Args:
    - data_file (str): Path to the preprocessed data CSV.
    - models_folder (str): Folder where trained models are stored.
    - scaler_file (str): Path to the pre-trained MinMaxScaler file.
    - output_file (str): Path to save the predicted data CSV.
    """
    

    
    # ✅ Step 1: Load the preprocessed property data
    try:
        data = pd.read_csv(data_file)
        print(f"✅ Loaded preprocessed data from {data_file}.")
    except FileNotFoundError:
        print(f"🚨 Data file not found: {data_file}")
        return
    
    # ✅ Step 2: Load trained models for each target variable
    target_models = {
        'Total ROI (%)': 'Total_ROI_(%).pkl',
        'Net Operating Income (NOI)': 'Net_Operating_Income.pkl',
        'Cap Rate (%)': 'Cap_Rate_(%).pkl',
        'Annual Cash Flow': 'Annual_Cash_Flow.pkl'
    }

    models = {}
    for target, model_file in target_models.items():
        model_path = os.path.join(models_folder, model_file)
        if os.path.exists(model_path):
            models[target] = joblib.load(model_path)
            print(f"✅ Loaded model: {target} from {model_path}")
        else:
            print(f"🚨 Model file not found: {model_path}")
            return

    # ✅ Step 3: Load the pre-trained scaler
    try:
        scaler = joblib.load(scaler_file)
        print(f"✅ Loaded scaler from {scaler_file}.")
    except FileNotFoundError:
        print(f"🚨 Scaler file not found: {scaler_file}")
        return

    # ✅ Step 4: Extract feature names from training
    feature_file = os.path.join(models_folder, "feature_names.pkl")
    try:
        feature_names = joblib.load(feature_file)
        print(f"✅ Loaded feature names from {feature_file}.")
    except FileNotFoundError:
        print(f"🚨 Feature names file not found: {feature_file}")
        return
    
    # ✅ Step 5: Ensure data has the necessary features
    missing_features = [feature for feature in feature_names if feature not in data.columns]
    if missing_features:
        print(f"🚨 Missing required features: {missing_features}")
        return

    # ✅ Step 6: Make predictions using each model
    print("🎯 Making predictions...")
    predictions = pd.DataFrame()
    for target, model in models.items():
        predictions[target] = model.predict(data[feature_names])

    # ✅ Step 7: Convert predictions back to real values using inverse transformation
    target_columns = list(models.keys())

    # Extract min-max scaled values for targets from the scaler
    min_max_params = scaler.data_min_, scaler.data_max_
    min_values, max_values = min_max_params

    for i, target in enumerate(target_columns):
        if target in scaler.feature_names_in_:  # Ensure the target was scaled
            min_val = min_values[i]
            max_val = max_values[i]
            predictions[target] = predictions[target] * (max_val - min_val) + min_val  # Inverse transform

    # ✅ Step 8: Combine Predictions with Input Data
    final_output = pd.concat([data, predictions], axis=1)

    # ✅ Step 9: Save Predictions to a CSV File
    final_output.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")

if __name__ == "__main__":
    make_predictions(
        data_file="../data/processed_prediction_data.csv",  # Preprocessed new data
        models_folder="../models/",                         # Folder containing trained models
        scaler_file="../models/scaler.pkl",                # Pre-trained scaler for inverse transform
        output_file="../data/predicted_real_values.csv"    # Output file
    )
