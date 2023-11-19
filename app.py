# Flask app

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Preprocess the input data
    age = data["age"]
    sex = data["sex"]
    bmi = data["bmi"]
    children = data["children"]
    smoker = data["smoker"]
    region_northwest = data["region_northwest"]
    region_southeast = data["region_southeast"]
    region_southwest = data["region_southwest"]

    # Make a prediction
    input_features = np.array(
        [
            [
                age,
                sex,
                bmi,
                children,
                smoker,
                region_northwest,
                region_southeast,
                region_southwest,
            ]
        ]
    )
    prediction = model.predict(input_features)

    # Format the prediction as JSON and return
    output = {"prediction": prediction.tolist()}
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=5000)
