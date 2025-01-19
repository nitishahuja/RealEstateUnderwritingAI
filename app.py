from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/cash_flow_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Prepare input for prediction
    features = [
        data["ZIP_Code"],
        data["Property_Type"],
        data["Square_Footage"],
        data["Asking_Price"],
        data["Estimated_Rent"],
        data["Vacancy_Rate"],
        data["NOI"],
        data["Cap_Rate"]
    ]
    # Predict
    prediction = model.predict([features])
    return jsonify({"predicted_cash_flow": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
