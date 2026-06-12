"""
STEP 1 of the live dashboard: train the Random Forest and save it to disk.

Run this ONCE before starting the app:
    python train_model.py

It reproduces the exact preprocessing from the capstone project
(same cleaning, features, split and seed), trains the best model
(Random Forest), prints its test accuracy, and saves everything the
web app needs into model.pkl.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "star_classification.csv"   # put the dataset next to this script

# ---- 1. identical preprocessing to the capstone notebook ----
df = pd.read_csv(DATA_PATH)

# remove the SDSS error code -9999 (failed measurements)
bad = (df[["u", "g", "r", "i", "z"]] == -9999).any(axis=1)
df = df[~bad]

# engineered colour indices
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]

FEATURES = ["u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z"]
X = df[FEATURES]
le = LabelEncoder()
y = le.fit_transform(df["class"])    # GALAXY=0, QSO=1, STAR=2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---- 2. train the best model from the comparison: Random Forest ----
print("Training Random Forest (100 trees) ...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print(f"Test accuracy: {acc:.4f}   macro F1: {f1:.4f}")

# ---- 3. save model + scaler + label encoder together ----
joblib.dump({
    "model": model,
    "scaler": scaler,
    "classes": list(le.classes_),
    "features": FEATURES,
    "test_accuracy": round(float(acc), 4),
    "test_f1": round(float(f1), 4),
}, "model.pkl", compress=3)
print("Saved model.pkl - now run:  python app.py")
