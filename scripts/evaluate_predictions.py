import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate_predictions(data_file, model_file, feature_file, scaler_file):
    try:
        # Load preprocessed data
        data = pd.read_csv(data_file)
        print("Loaded data for evaluation.")

        # Load the trained model
        model = joblib.load(model_file)
        print("Loaded trained model.")

        # Load the feature names
        feature_names = joblib.load(feature_file)
        print("Loaded feature names.")

        # Load the scaler
        scaler = joblib.load(scaler_file)
        print("Loaded scaler for reverse transformation.")

        # Select features for prediction
        features = data[feature_names]

        # Make predictions
        predictions = model.predict(features)

        # Reverse-transform predictions to raw values
        target_columns = ["Total ROI (%)", "Net Operating Income (NOI)", "Cap Rate (%)", "Annual Cash Flow"]

        if hasattr(scaler, 'mean_') and len(scaler.mean_) >= len(target_columns):
            # Slice the scaler to match the target columns
            target_scaler = scaler  # Replace with exact scaler for targets, if saved separately
            raw_predictions = predictions * target_scaler.scale_[:len(target_columns)] + target_scaler.mean_[:len(target_columns)]
            for i, column in enumerate(target_columns):
                data[column] = raw_predictions[:, i]
            print("Reverse-transformed predictions to raw values.")
        else:
            print("Scaler is not correctly configured for the targets.")
            return

        # Save the predictions to a new CSV file
        output_file = data_file.replace("processed", "evaluated")
        data.to_csv(output_file, index=False)
        print(f"Evaluated predictions saved to {output_file}.")

        # Evaluate the predictions against true targets
        if set(target_columns).issubset(data.columns):
            true_targets = data[target_columns]
            mse = mean_squared_error(true_targets, raw_predictions, multioutput="raw_values")
            mae = mean_absolute_error(true_targets, raw_predictions, multioutput="raw_values")
            r2 = r2_score(true_targets, raw_predictions, multioutput="raw_values")

            print("\nEvaluation Metrics:")
            for i, target in enumerate(target_columns):
                print(f"Target: {target}")
                print(f"  - MSE: {mse[i]:.4f}")
                print(f"  - MAE: {mae[i]:.4f}")
                print(f"  - R²: {r2[i]:.4f}")

            # Plot actual vs predicted values
            print("\nGenerating actual vs predicted plots...")
            for i, target in enumerate(target_columns):
                plt.figure()
                plt.scatter(true_targets.iloc[:, i], raw_predictions[:, i], alpha=0.5)
                plt.plot(
                    [true_targets.iloc[:, i].min(), true_targets.iloc[:, i].max()],
                    [true_targets.iloc[:, i].min(), true_targets.iloc[:, i].max()],
                    color='red', linestyle='--'
                )
                plt.title(f"Actual vs. Predicted: {target}")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.show()
        else:
            print("True target values not available for comparison.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    evaluate_predictions(
        data_file="../data/processed_new_property_data.csv",  # Processed data file for prediction
        model_file="../models/underwriting_model.pkl",        # Trained model
        feature_file="../models/feature_names.pkl",           # Feature names
        scaler_file="../models/scaler.pkl"                    # Scaler for reverse transformation
    )
