from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("loan_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [[
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Credit_History"]
    ]]

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": "Loan Approved" if prediction == 1 else "Loan Rejected"
    })

if __name__ == "__main__":
    app.run()
