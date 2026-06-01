from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["vitamin_c"]),
            float(request.form["vitamin_b11"]),
            float(request.form["calcium"]),
            float(request.form["calories"]),
            float(request.form["protein"]),
            float(request.form["carbs"]),  # Changed from carbohydrates to carbs
            float(request.form["sugars"]),
            float(request.form["fat"]),
            float(request.form["iron"]),
            float(request.form["fiber"]),
            float(request.form["sodium"])
        ]

        data = np.array([features])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)  # Use scaled data

        if prediction[0] == 2:
            result = "Healthy ✅"
        elif prediction[0] == 1:
            result = "Moderate ⚠️"
        else:
            result = "Unhealthy ❌"

        return render_template("index.html", prediction=result)
    
    except KeyError as e:
        return f"Missing form field: {e}", 400
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)