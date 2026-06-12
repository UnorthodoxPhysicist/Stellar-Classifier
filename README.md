# Stellar Classifier — Live Prediction App

The trained Random Forest (97.97% test accuracy) served by a Flask backend,
with a web UI for classifying objects on the spot — manually or fetched
live from the Sloan Digital Sky Survey.

## Run it

1. Put `star_classification.csv` in this folder.
2. Install dependencies:   `pip install -r requirements.txt`
3. Train & save the model (once, ~1 min):   `python train_model.py`
4. Start the app:   `python app.py`
5. Open  http://127.0.0.1:5000

## How it works

- `train_model.py` reproduces the exact capstone preprocessing (cleaning,
  colour indices, 80/20 stratified split, StandardScaler) and saves the
  fitted Random Forest + scaler + label encoder into `model.pkl`.
- `app.py` loads `model.pkl` once at startup and exposes:
  - `POST /predict` — takes u, g, r, i, z, redshift; computes the colour
    indices server-side, scales, and returns the class + per-class
    probabilities (the fraction of the 100 trees voting for each class).
  - `POST /sdss_lookup` — queries the SDSS DR17 SkyServer API for the
    nearest spectroscopic object to a given RA/Dec and returns its real
    photometry, redshift, and SDSS's own class label, so the model's
    prediction can be compared against the survey's ground truth.
- `templates/index.html` — the frontend UI (no framework, plain JS).

Try RA = 184.9511, Dec = −0.8001 for a real SDSS galaxy.

## Deploy publicly (Render, free)

1. Push this folder to a GitHub repository (model.pkl included — it's only 6.5 MB,
   so the server never needs the dataset or retraining).
2. On https://render.com → New → Web Service → connect the repo.
3. Settings: Build command `pip install -r requirements.txt`,
   Start command `gunicorn app:app`, Instance type **Free**.
4. Deploy — your app gets a public URL like https://stellar-classifier.onrender.com

Note: free instances sleep after ~15 min of inactivity; the first visit
afterwards takes ~50 s to wake up.
