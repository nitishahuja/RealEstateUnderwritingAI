import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def evaluate_model(data_file, models_folder, feature_file, results_folder):
    """
    Evaluates trained models for Total ROI (%), Net Operating Income (NOI), 
    Cap Rate (%), and Annual Cash Flow.
    
    - Loads the trained models from `models_folder`
    - Evaluates them using preprocessed data from `data_file`
    - Saves evaluation results to `results_folder`
    """

    # ✅ Ensure results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # ✅ Load preprocessed data
    try:
        data = pd.read_csv(data_file)
        print("✅ Loaded data for evaluation.")
    except FileNotFoundError:
        raise ValueError(f"🚨 Data file not found: {data_file}")

    # ✅ Load feature names
    try:
        feature_names = joblib.load(feature_file)
    except FileNotFoundError:
        raise ValueError(f"🚨 Feature file not found: {feature_file}")

    # ✅ Define target variables & corresponding model files
    targets = {
        "Total ROI (%)": "Total_ROI_(%).pkl",
        "Net Operating Income (NOI)": "Net_Operating_Income.pkl",
        "Cap Rate (%)": "Cap_Rate_(%).pkl",
        "Annual Cash Flow": "Annual_Cash_Flow.pkl",
    }

    # ✅ Ensure all features exist in data
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        raise ValueError(f"🚨 Missing features in data: {missing_features}")

    # ✅ Extract features for prediction
    X = data[feature_names]

    # ✅ Initialize results storage
    results = []

    # ✅ Evaluate each model separately
    for target, model_file in targets.items():
        model_path = os.path.join(models_folder, model_file)

        # 🔄 Load model
        try:
            model = joblib.load(model_path)
            print(f"\n🎯 Evaluating Model: {target}")
        except FileNotFoundError:
            print(f"🚨 Model file not found: {model_path}")
            continue

        # ✅ Ensure target exists in data
        if target not in data.columns:
            print(f"🚨 Missing target variable in data: {target}")
            continue

        # ✅ Extract actual target values
        y_actual = data[target]

        # 🔮 Generate predictions
        y_pred = model.predict(X)

        # 📊 Compute evaluation metrics
        mae = mean_absolute_error(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, y_pred)
        
        # 📈 Estimate model accuracy in simple terms
        avg_actual = y_actual.mean()
        accuracy = max(0, 100 - (rmse / avg_actual * 100))  # Ensures it doesn’t go negative
        
        # 📊 Print evaluation results
        print(f"📊 Model Performance for {target}:")
        print(f"✅ Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"✅ Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ Model Accuracy: {accuracy:.2f}%")

        # ✅ Store results for later saving
        results.append({"Target": target, "MAE": mae, "RMSE": rmse, "R²": r2, "Accuracy (%)": accuracy})

        # 📊 Generate visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(y_actual, y_pred, alpha=0.5, label="Predictions")
        plt.plot(
            [y_actual.min(), y_actual.max()],
            [y_actual.min(), y_actual.max()],
            color="red",
            linestyle="--",
            label="Ideal Fit",
        )
        plt.title(f"Actual vs Predicted: {target}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid(True)

        # ✅ Save plot
        plot_path = os.path.join(results_folder, f"actual_vs_predicted_{target.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        print(f"📊 Saved plot: {plot_path}")
        plt.close()

    # ✅ Save evaluation metrics to a file

    # ✅ Save evaluation metrics to a file
    results_df = pd.DataFrame(results)
    results_file = os.path.join(results_folder, "evaluation_metrics.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Evaluation metrics saved to {results_file}.")



if __name__ == "__main__":
    evaluate_model(
        data_file="../data/processed_training_data.csv",   # Preprocessed data file
        models_folder="../models/",                       # Folder containing trained models
        feature_file="../models/feature_names.pkl",       # Feature names file
        results_folder="../results/"                      # Where to save results
    )
