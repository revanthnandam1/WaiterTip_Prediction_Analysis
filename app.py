from flask import Flask, render_template, request
import pandas as pd
from scripts.preprocess import preprocess_data
from scripts.train_model import train_model

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv("data/tips.csv")
preprocessed_data, _ = preprocess_data(data)

# Train the model
model = train_model(preprocessed_data)

@app.route("/")
def index():
    """Render the index page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and display results."""
    # Get user input from the form
    total_bill = float(request.form["total_bill"])
    sex = request.form["sex"]
    smoker = request.form["smoker"]
    day = request.form["day"]
    time = request.form["time"]

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        "total_bill": [total_bill],
        "sex": [sex],
        "smoker": [smoker],
        "day": [day],
        "time": [time]
    })

    # Preprocess the input data
    input_data, _ = preprocess_data(input_data)

    # Make a prediction
    tip_prediction = model.predict(input_data)[0]

    # Render the result page with the prediction
    return render_template("result.html", prediction=tip_prediction)

if __name__ == "__main__":
    app.run(debug=True)