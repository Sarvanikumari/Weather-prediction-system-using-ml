# ==============================================================
#  retrain_model.py  —  Run this ONCE to fix your model
#  It saves a Pipeline (StandardScaler + SVC) as model.pkl
#  so your Flask app will work correctly forever.
# ==============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. Load your training data ─────────────────────────────────────────────────
#  Replace 'your_dataset.csv' with your actual CSV file name.
#  The CSV must have these 10 columns + a target column (RainTomorrow: 1 or 0).
df = pd.read_csv("your_dataset.csv")

FEATURE_COLS = [
    "min_temp", "max_temp", "rainfall", "evaporation", "sunshine",
    "wind_speed9", "wind_speed3", "humidity9", "humidity3", "pressure9"
]
TARGET_COL = "RainTomorrow"   # change to your actual target column name

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# ── 2. Train/test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Build a Pipeline: Scaler → SVC ─────────────────────────────────────────
#  The Pipeline scales inputs automatically — no manual scaling needed ever again.
pipeline = Pipeline([
    ("scaler", StandardScaler()),    # scales each feature to mean=0, std=1
    ("svc",    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
])

# ── 4. Train ───────────────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ── 5. Evaluate ────────────────────────────────────────────────────────────────
y_pred   = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"✅ Model Accuracy: {accuracy:.2f}%")

# ── 6. Save the FULL Pipeline as model.pkl ────────────────────────────────────
#  Now model.pkl contains BOTH the scaler and the SVC.
#  Your Flask app can call model.predict(raw_input) directly — no preprocessing!
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Saved Pipeline (Scaler + SVC) to model.pkl")
print("   Your Flask app now works correctly without any manual scaling.")

# ── 7. Quick sanity check ──────────────────────────────────────────────────────
rain_input   = np.array([[10.0, 18.0, 12.0, 3.0, 1.0, 30, 35, 95, 90, 998.0]])
norain_input = np.array([[20.0, 34.0,  0.0, 9.0, 12.0, 10, 12, 32, 20, 1025.0]])

print(f"\n🔍 Sanity check:")
print(f"   Rain-like input    → {'Rain' if pipeline.predict(rain_input)[0]==1 else 'No Rain'}")
print(f"   No-rain-like input → {'Rain' if pipeline.predict(norain_input)[0]==1 else 'No Rain'}")
