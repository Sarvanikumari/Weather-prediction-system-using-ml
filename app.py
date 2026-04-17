# ==============================================================
#  app.py  —  Fixed Flask Rainfall Prediction App
#  Works correctly whether model.pkl is:
#    (a) a Pipeline  (scaler + SVC) — recommended after retraining
#    (b) raw SVC only               — applies manual scaling as fallback
# ==============================================================

from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "rainfall_secret_key_2024"

# ── Load model ─────────────────────────────────────────────────────────────────
with open(r"C:\Users\sarva\OneDrive\Desktop\weather prediction\weather\model.pkl", "rb") as f:
    model = pickle.load(f)

# Detect model type
IS_PIPELINE = hasattr(model, "steps")
print(f"Model type: {'Pipeline (Scaler+SVC) ✅' if IS_PIPELINE else 'Raw SVC (manual scaling needed)'}")

# ── Fallback scaler stats (only used if model is raw SVC, NOT a Pipeline) ──────
#  These are approximated from the weatherAUS dataset.
#  They may not match your exact training data — RETRAINING IS THE PROPER FIX.
FEATURE_MEANS = np.array([12.19, 23.22,  2.36,  5.47,  7.61,
                           14.04, 18.66, 68.88, 51.54, 1017.65])
FEATURE_STDS  = np.array([ 6.40,  7.12,  8.08,  4.19,  3.78,
                            8.92,  8.84, 19.03, 20.53,    7.11])

MODEL_ACCURACY = 92.5
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# 10 features in the exact order the model was trained on
FEATURES = [
    {"label": "Min Temperature (°C)",      "name": "min_temp",    "placeholder": "e.g. 13.0"},
    {"label": "Max Temperature (°C)",      "name": "max_temp",    "placeholder": "e.g. 25.0"},
    {"label": "Rainfall (mm)",             "name": "rainfall",    "placeholder": "e.g. 2.0"},
    {"label": "Evaporation (mm)",          "name": "evaporation", "placeholder": "e.g. 5.5"},
    {"label": "Sunshine (hrs)",            "name": "sunshine",    "placeholder": "e.g. 7.0"},
    {"label": "Wind Speed 9am (km/h)",     "name": "wind_speed9", "placeholder": "e.g. 14"},
    {"label": "Wind Speed 3pm (km/h)",     "name": "wind_speed3", "placeholder": "e.g. 19"},
    {"label": "Humidity 9am (%)",          "name": "humidity9",   "placeholder": "e.g. 70"},
    {"label": "Humidity 3pm (%)",          "name": "humidity3",   "placeholder": "e.g. 52"},
    {"label": "Pressure 9am (hPa)",        "name": "pressure9",   "placeholder": "e.g. 1018"},
]

def make_prediction(raw_values):
    """
    Takes raw input values from the form and returns 'Rain' or 'No Rain'.
    Automatically handles scaling based on model type.
    """
    arr = np.array([raw_values])          # shape: (1, 10)

    if IS_PIPELINE:
        # Pipeline scales internally — just pass raw values directly ✅
        result = model.predict(arr)[0]
    else:
        # Raw SVC — must manually scale before predicting ⚠️
        arr_scaled = (arr - FEATURE_MEANS) / FEATURE_STDS
        result = model.predict(arr_scaled)[0]

    return "🌧️ Rain Expected" if result == 1 else "☀️ No Rain Expected"


# ── LOGIN ──────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if (request.form.get("username") == VALID_USERNAME and
                request.form.get("password") == VALID_PASSWORD):
            session["logged_in"] = True
            session["username"]  = request.form.get("username")
            return redirect(url_for("home"))
        error = "Invalid username or password. Please try again."
    return render_template("login.html", error=error)


# ── HOME ───────────────────────────────────────────────────────────────────────
@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("home.html",
                           accuracy=MODEL_ACCURACY,
                           username=session.get("username"))


# ── PREDICT ────────────────────────────────────────────────────────────────────
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


# ── LOGOUT ─────────────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)