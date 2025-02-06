import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

# 🔧 Debug Mode: Train on a small dataset first
DEBUG_MODE = False  # Set to False to train on full data

def train_model(data_file, model_output_folder, feature_output):
    """
    Trains separate LightGBM regression models for each target variable.
    - Uses hyperparameter tuning
    - Saves each model separately
    - Logs performance metrics
    """

    # ✅ Step 1: Load Data
    try:
        data = pd.read_csv(data_file)
        print("✅ Loaded preprocessed training data.")
    except FileNotFoundError:
        raise ValueError(f"🚨 Data file not found: {data_file}")

    # 🔧 Debug Mode: Use only 5,000 rows for quick testing
    if DEBUG_MODE:
        data = data.sample(n=5000, random_state=42)
        print("⚡ DEBUG MODE: Using only 5,000 rows for quick testing.")

    # ✅ Step 2: Load Features from Preprocessing
    try:
        feature_names = joblib.load(feature_output)
    except FileNotFoundError:
        raise ValueError(f"🚨 Feature names file not found: {feature_output}")

    target_names = [
        'Total ROI (%)', 'Net Operating Income (NOI)', 'Cap Rate (%)', 'Annual Cash Flow'
    ]

    # ✅ Step 3: Check for Missing Columns
    missing_features = [col for col in feature_names if col not in data.columns]
    missing_targets = [col for col in target_names if col not in data.columns]
    if missing_features or missing_targets:
        raise ValueError(f"🚨 Missing columns in data. Features: {missing_features}, Targets: {missing_targets}")

    # ✅ Step 4: Define Features
    X = data[feature_names]

    # ✅ Step 5: Ensure Model Output Folder Exists
    os.makedirs(model_output_folder, exist_ok=True)

    # ✅ Step 6: Train Separate Models for Each Target Variable
    trained_models = []  # Track trained models

    for target in target_names:
        print(f"\n🎯 Training model for {target}...")

        # ✅ Step 7: Define Target Variable
        y = data[target]

        # ✅ Step 8: Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ✅ Step 9: Hyperparameter Tuning
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [-1, 3, 5],
            "num_leaves": [31, 50, 100],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 2, 5]
        }

        lgbm = LGBMRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_grid,
            n_iter=5,  # Reduce iterations for speed
            scoring="neg_mean_squared_error",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        print(f"🔍 Running Hyperparameter Tuning for {target}...")
        random_search.fit(X_train, y_train)

        # ✅ Step 10: Train Best Model
        best_params = random_search.best_params_
        print(f"✅ Best Parameters for {target}: {best_params}")

        best_model = LGBMRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        # ✅ Step 11: Save the Model
        model_path = os.path.join(model_output_folder, f"{target.replace(' ', '_')}.pkl")
        joblib.dump(best_model, model_path)
        trained_models.append(target)  # Keep track of trained models
        print(f"✅ Model for {target} saved to {model_path}")

        # ✅ Step 12: Evaluate Model Performance
        y_test_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        avg_actual = y_test.mean()  # Mean of actual values
        accuracy = 100 - (rmse / avg_actual * 100)  # Approximate accuracy

        print(f"\n📊 Model Performance for {target}:")
        print(f"✅ Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"✅ Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"✅ Model Accuracy: {accuracy:.2f}%")

        # ✅ Step 13: Visualize Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, best_model.feature_importances_)
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Features")
        plt.title(f"Feature Importance for {target}")
        plt.grid(True)
        plt.show()

    # ✅ Step 14: Final Check - Ensure All Models Were Saved
    print("\n🎯 Final Check: Models Trained & Saved")
    for target in target_names:
        model_file = os.path.join(model_output_folder, f"{target.replace(' ', '_')}.pkl")
        if os.path.exists(model_file):
            print(f"✅ Model for {target} exists: {model_file}")
        else:
            print(f"🚨 ERROR: Model for {target} is MISSING!")

if __name__ == "__main__":
    train_model(
        "../data/processed_training_data.csv", 
        "../models", 
        "../models/feature_names.pkl"
    )
