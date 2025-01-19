import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import numpy as np


def train_model(data_file, model_output, feature_output):
    # Load preprocessed data
    data = pd.read_csv(data_file)
    print("Loaded preprocessed training data.")

    # Define features and targets
    feature_names = [
        'Neighborhood', 'Property Type', 'Square Footage', 'Purchase Price',
        'Current Market Rent', 'Vacancy Rate', 'Net Operating Income (NOI)',
        'Rental Yield', 'Debt to Income Ratio', 'Property Age'
    ]
    target_names = [
        'Total ROI (%)', 'Net Operating Income (NOI)', 'Cap Rate (%)', 'Annual Cash Flow'
    ]

    # Check if features and targets exist in the data
    missing_features = [col for col in feature_names if col not in data.columns]
    missing_targets = [col for col in target_names if col not in data.columns]
    if missing_features or missing_targets:
        raise ValueError(f"Missing columns in data. Features: {missing_features}, Targets: {missing_targets}")

    features = data[feature_names]
    targets = data[target_names]

    # Train-test split
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 2, 5]
    }
    xgb = XGBRegressor(random_state=42)
    print("Tuning hyperparameters...")
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train, y_train)

    # Save the best model
    model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")
    joblib.dump(model, model_output)
    print(f"Model saved to {model_output}")

    # Save feature names
    joblib.dump(feature_names, feature_output)
    print(f"Feature names saved to {feature_output}")

    # Evaluate on test set
    print("Evaluating model on test data...")
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_test_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_test, y_test_pred, multioutput="raw_values")
    r2 = r2_score(y_test, y_test_pred, multioutput="raw_values")

    print("\nTest Performance:")
    for i, target_name in enumerate(target_names):
        print(f"Target: {target_name}")
        print(f"  - MSE: {mse[i]:.4f}")
        print(f"  - MAE: {mae[i]:.4f}")
        print(f"  - R²: {r2[i]:.4f}")

    # Visualize actual vs predicted
    print("\nGenerating actual vs. predicted plots...")
    for i, target_name in enumerate(target_names):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test.iloc[:, i], y_test_pred[:, i], alpha=0.5)
        plt.plot(
            [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
            [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
            color='red', linestyle='--'
        )
        plt.title(f"Actual vs. Predicted for {target_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        plt.show()

    # Visualize feature importance
    print("\nGenerating feature importance plot...")
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, model.feature_importances_)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_model(
        "../data/processed_training_data.csv",  # Input file
        "../models/underwriting_model.pkl",     # Model output
        "../models/feature_names.pkl"           # Feature names output
    )
