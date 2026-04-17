# ==============================================================
#  app.py — Fixed Flask Rainfall Prediction App (Render Ready)
# ==============================================================

from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "rainfall_secret_key_2024"

# ── Load model (FIXED PATH) ────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Detect model type
IS_PIPELINE = hasattr(model, "steps")
print(f"Model type: {'Pipeline (Scaler+SVC)' if IS_PIPELINE else 'Raw SVC'}")

# ── Fallback scaler stats ─────────────────────────────────────
FEATURE_MEANS = np.array([12.19, 23.22, 2.36, 5.47, 7.61,
                          14.04, 18.66, 68.88, 51.54, 1017.65])

FEATURE_STDS  = np.array([6.40, 7.12, 8.08, 4.19, 3.78,
                          8.92, 8.84, 19.03, 20.53, 7.11])

MODEL_ACCURACY = 92.5
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

FEATURES = [
    {"label": "Min Temperature (°C)", "name": "min_temp"},
    {"label": "Max Temperature (°C)", "name": "max_temp"},
    {"label": "Rainfall (mm)", "name": "rainfall"},
    {"label": "Evaporation (mm)", "name": "evaporation"},
    {"label": "Sunshine (hrs)", "name": "sunshine"},
    {"label": "Wind Speed 9am (km/h)", "name": "wind_speed9"},
    {"label": "Wind Speed 3pm (km/h)", "name": "wind_speed3"},
    {"label": "Humidity 9am (%)", "name": "humidity9"},
    {"label": "Humidity 3pm (%)", "name": "humidity3"},
    {"label": "Pressure 9am (hPa)", "name": "pressure9"},
]

# ── Prediction function ───────────────────────────────────────
def make_prediction(raw_values):
    arr = np.array([raw_values])

    if IS_PIPELINE:
        result = model.predict(arr)[0]
    else:
        arr_scaled = (arr - FEATURE_MEANS) / FEATURE_STDS
        result = model.predict(arr_scaled)[0]

    return "🌧️ Rain Expected" if result == 1 else "☀️ No Rain Expected"


# ── LOGIN ─────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if (request.form.get("username") == VALID_USERNAME and
            request.form.get("password") == VALID_PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password"
    return render_template("login.html", error=error)


# ── HOME ──────────────────────────────────────────────────────
@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("home.html", accuracy=MODEL_ACCURACY)


# ── PREDICT ───────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    prediction = None
    if request.method == "POST":
        try:
            values = [float(request.form.get(f["name"], 0)) for f in FEATURES]
            prediction = make_prediction(values)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, features=FEATURES)


# ── LOGOUT ────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── RUN (IMPORTANT FOR RENDER) ─────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
