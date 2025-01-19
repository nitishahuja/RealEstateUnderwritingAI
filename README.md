# **RealEstateUnderwritingAI**

An AI-powered tool designed for real estate underwriting to predict key investment metrics such as ROI (Return on Investment), NOI (Net Operating Income), Cap Rate, and Annual Cash Flow. This project uses synthetic data generation, advanced preprocessing, and machine learning to help investors make data-driven property decisions.

---

## **Features**

- **Realistic Data Simulation**: Generate realistic real estate data for training and testing the model.
- **Preprocessing Pipeline**: Clean, validate, and preprocess raw property data for model training.
- **Predictive Model**: Train a machine learning model to predict investment metrics with high accuracy.
- **Evaluation Metrics**: Analyze model performance using metrics like MSE (Mean Squared Error) and R².
- **Prediction Pipeline**: Input new property data to predict investment metrics in real-time.

---

## **Tech Stack**

- **Programming Language**: Python
- **Libraries & Tools**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Joblib
- **Data**: Synthetic and realistic property data, designed to mimic Boston neighborhoods.

---

## **Getting Started**

Follow these steps to set up the project and run the pipelines.

### **Prerequisites**

1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RealEstateUnderwritingAI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd RealEstateUnderwritingAI
   ```

---

## **Workflow**

### 1. **Data Generation**

Use the script to generate synthetic property data for model training.

```bash
python data_generation.py
```

Output: `raw_training_data.csv`

### 2. **Data Preprocessing**

Clean and preprocess raw data for model training and predictions.

```bash
python preprocess.py
```

Output: `processed_training_data.csv` for training or `processed_new_property_data.csv` for predictions.

### 3. **Model Training**

Train the XGBoost model on the processed data.

```bash
python train_model.py
```

Output: `underwriting_model.pkl` and `scaler.pkl`

### 4. **Model Evaluation**

Evaluate the trained model for performance on test data.

```bash
python evaluate_model.py
```

### 5. **Prediction**

Predict key metrics for new property data.

```bash
python predict.py
```

Output: A CSV file with predictions.

---

## **Key Components**

### **1. Data Generation**

Generates synthetic property data for realistic training:

- Includes key features like neighborhood, property type, financial details, and derived metrics.

### **2. Data Preprocessing**

- Handles missing values, outliers, and scales numeric features.
- Encodes categorical features for model compatibility.
- Outputs clean data ready for training or predictions.

### **3. Machine Learning Model**

- **Model Used**: XGBoost for multi-output regression.
- Predicts:
  - **Net Operating Income (NOI)**
  - **Cap Rate (%)**
  - **Annual Cash Flow**
  - **Total ROI (%)**

### **4. Evaluation**

- Evaluates model using:
  - **MSE (Mean Squared Error)**: Measures average squared difference between predicted and actual values.
  - **R² Score**: Represents the proportion of variance captured by the model.

### **5. Prediction**

- Accepts new property data.
- Outputs predicted metrics for underwriting decisions.

---

## **File Structure**

```
RealEstateUnderwritingAI/
│
├── data/                         # Contains generated and processed datasets
│   ├── raw_training_data.csv
│   ├── processed_training_data.csv
│   ├── processed_new_property_data.csv
│
├── models/                       # Contains trained models and scalers
│   ├── underwriting_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│
├── scripts/                      # Core scripts for the workflow
│   ├── data_generation.py        # Generates synthetic data
│   ├── preprocess.py             # Preprocessing pipeline
│   ├── train_model.py            # Model training
│   ├── evaluate_model.py         # Model evaluation
│   ├── predict.py                # Prediction script
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

---

## **Usage Examples**

### **Prediction Example**

Input:

```json
{
  "Neighborhood": "South Boston",
  "Property Type": "Single-family",
  "Year Built": 2015,
  "Square Footage": 2500,
  "Bedrooms": 4,
  "Bathrooms": 3,
  "Purchase Price": 500000,
  "Down Payment": 100000,
  "Loan Amount": 400000,
  "Loan Term (Years)": 30,
  "Interest Rate (%)": 3.8,
  "Property Taxes": 10000,
  "Insurance Costs": 2000,
  "HOA Fees": 500,
  "Maintenance Costs": 3000,
  "Renovation Costs": 15000,
  "Current Market Rent": 4000,
  "Vacancy Rate": 0.05,
  "Rental Growth Rate (%)": 0.03,
  "Appreciation Rate (%)": 0.02
}
```

Output:

```json
{
  "Net Operating Income (NOI)": 25000,
  "Cap Rate (%)": 5.0,
  "Annual Cash Flow": 12000,
  "Total ROI (%)": 180.5
}
```

---

## **Contributing**

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch-name
   ```
3. Commit changes:
   ```bash
   git commit -m "Your commit message"
   ```
4. Push the branch:
   ```bash
   git push origin feature-branch-name
   ```
5. Submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
