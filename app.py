from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Load the model and target names
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

target = list(np.load("features.npy"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Create a DataFrame from input data
        input_data = pd.DataFrame([{
            "sepal length (cm)": data.get("sepal length (cm)"),
            "sepal width (cm)": data.get("sepal width (cm)"),
            "petal length (cm)": data.get("petal length (cm)"),
            "petal width (cm)": data.get("petal width (cm)")
        }])

        # Validate input (ensure no missing or None values)
        if input_data.isnull().values.any():
            return jsonify({"error": "Missing feature values"}), 400

        # Predict using the model
        prediction = model.predict(input_data)

        # Map prediction to target class
        predicted_class = target[prediction[0]]
        return jsonify({"Prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True,port=6900)
