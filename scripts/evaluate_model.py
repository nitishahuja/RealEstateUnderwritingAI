import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def evaluate_model(data_file, model_file, feature_file):
    # Load preprocessed data
    try:
        data = pd.read_csv(data_file)
        print("Loaded data for evaluation.")
    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        return

    # Load feature names
    try:
        feature_names = joblib.load(feature_file)
    except FileNotFoundError:
        print(f"Feature file not found: {feature_file}")
        return

    # Check if features exist in the data
    missing_features = [feature for feature in feature_names if feature not in data.columns]
    if missing_features:
        print(f"Missing features in data: {missing_features}")
        return

    # Select features
    features = data[feature_names]

    # Load the model
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        print(f"Model file not found: {model_file}")
        return

    # Define target variables
    target_names = [
        'Total ROI (%)', 'Net Operating Income (NOI)', 'Cap Rate (%)', 'Annual Cash Flow'
    ]

    # Check if targets exist in the data
    missing_targets = [target for target in target_names if target not in data.columns]
    if missing_targets:
        print(f"Missing target variables in data: {missing_targets}")
        return

    # Select targets
    targets = data[target_names]

    # Predict and evaluate
    print("Generating predictions...")
    predictions = model.predict(features)

    # Calculate metrics
    mse = mean_squared_error(targets, predictions, multioutput="raw_values")
    mae = mean_absolute_error(targets, predictions, multioutput="raw_values")
    r2 = r2_score(targets, predictions, multioutput="raw_values")

    # Print evaluation results
    print("\nEvaluation Results:")
    for i, target in enumerate(target_names):
        print(f"Target: {target}")
        print(f"  - MSE: {mse[i]:.4f}")
        print(f"  - MAE: {mae[i]:.4f}")
        print(f"  - R²: {r2[i]:.4f}")

    # Visualize actual vs predicted for each target variable
    print("\nGenerating actual vs. predicted plots...")
    for i, target in enumerate(target_names):
        plt.figure(figsize=(8, 6))
        plt.scatter(targets.iloc[:, i], predictions[:, i], alpha=0.5, label='Predictions')
        plt.plot(
            [targets.iloc[:, i].min(), targets.iloc[:, i].max()],
            [targets.iloc[:, i].min(), targets.iloc[:, i].max()],
            color='red', linestyle='--', label='Ideal Fit'
        )
        plt.title(f"Actual vs Predicted: {target}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save evaluation metrics to a file (optional)
    metrics_summary = pd.DataFrame({
        "Target": target_names,
        "MSE": mse,
        "MAE": mae,
        "R²": r2
    })
    metrics_summary.to_csv("../results/evaluation_metrics.csv", index=False)
    print("Evaluation metrics saved to '../results/evaluation_metrics.csv'.")


if __name__ == "__main__":
    evaluate_model(
        "../data/processed_training_data.csv",  # Data file
        "../models/underwriting_model.pkl",     # Model file
        "../models/feature_names.pkl"           # Feature names file
    )
