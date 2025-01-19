import pandas as pd
import joblib

def make_predictions(data_file, model_file, feature_file, output_file):
    try:
        # Load the preprocessed data
        data = pd.read_csv(data_file)
        print(f"Loaded preprocessed data from {data_file}.")
    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Data file is empty: {data_file}")
        return

    try:
        # Load the trained model
        model = joblib.load(model_file)
        print(f"Loaded trained model from {model_file}.")
    except FileNotFoundError:
        print(f"Model file not found: {model_file}")
        return

    try:
        # Load the feature names
        feature_names = joblib.load(feature_file)
        print(f"Loaded feature names from {feature_file}.")
    except FileNotFoundError:
        print(f"Feature file not found: {feature_file}")
        return

    # Ensure required features exist in the data
    missing_features = [feature for feature in feature_names if feature not in data.columns]
    if missing_features:
        print(f"Missing required features in the data: {missing_features}")
        return

    # Select the correct features for prediction
    features = data[feature_names]
    print("Selected features for prediction.")

    try:
        # Make predictions
        predictions = model.predict(features)
        print("Predictions generated successfully.")
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return

    # Add predictions to the original data
    target_columns = ['Total ROI (%)', 'Net Operating Income (NOI)', 'Cap Rate (%)', 'Annual Cash Flow']
    for i, target in enumerate(target_columns):
        data[target] = predictions[:, i]
    print("Added predictions to the original data.")

    try:
        # Save the results to a new file
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}.")
    except Exception as e:
        print(f"Error saving predictions to file: {e}")

if __name__ == "__main__":
    make_predictions(
        "../data/processed_new_property_data.csv",  # Preprocessed new data
        "../models/underwriting_model.pkl",        # Trained model
        "../models/feature_names.pkl",             # Feature names
        "../data/predicted_property_data.csv"      # Output file
    )
