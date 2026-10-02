from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model_bundle = joblib.load("models/house_price_model.pkl")

model = model_bundle["model"]
features = model_bundle["features"]
metrics = model_bundle["metrics"]


# Feature importance from the Random Forest model
feature_importance = sorted(
    zip(features, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)


# Supported input ranges
INPUT_RANGES = {
    "MedInc": (0.4999, 15.0001),
    "HouseAge": (1, 52),
    "AveRooms": (0.846, 141.91),
    "AveBedrms": (0.333, 34.07),
    "Population": (3, 35682),
    "AveOccup": (0.69, 1244),
    "Latitude": (32.54, 41.95),
    "Longitude": (-124.35, -114.31)
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from the form
        input_data = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
            float(request.form["Latitude"]),
            float(request.form["Longitude"])
        ]

    except (ValueError, KeyError):

        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )


    # Validate each input against supported ranges
    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    for name, value in zip(feature_names, input_data):

        minimum, maximum = INPUT_RANGES[name]

        if value < minimum or value > maximum:

            return render_template(
                "index.html",
                error=(
                    f"{name} must be between "
                    f"{minimum} and {maximum}."
                )
            )


    # Store property details for result page
    property_data = {
        "MedInc": input_data[0],
        "HouseAge": input_data[1],
        "AveRooms": input_data[2],
        "AveBedrms": input_data[3],
        "Population": input_data[4],
        "AveOccup": input_data[5],
        "Latitude": input_data[6],
        "Longitude": input_data[7]
    }


    # Make prediction
    prediction = model.predict([input_data])[0]


    # Convert from $100,000s to dollars
    price = prediction * 100000


    # Send result to result.html
    return render_template(
        "result.html",
        price=f"{price:,.2f}",
        metrics=metrics,
        feature_importance=feature_importance,
        property_data=property_data
    )


if __name__ == "__main__":
    app.run(debug=True)