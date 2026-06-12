"""
STEP 2 of the live dashboard: the Flask web app.

    python app.py          ->  open http://127.0.0.1:5000

Endpoints
---------
GET  /             the dashboard UI
POST /predict      JSON {u, g, r, i, z, redshift} -> Random Forest prediction
POST /sdss_lookup  JSON {ra, dec} -> fetches a real object live from the
                   SDSS SkyServer API (needs internet), returns its
                   photometry + redshift + SDSS's own class label
"""
import joblib
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---- load the trained model once at startup ----
bundle = joblib.load("model.pkl")
MODEL = bundle["model"]
SCALER = bundle["scaler"]
CLASSES = bundle["classes"]          # ['GALAXY', 'QSO', 'STAR']
FEATURES = bundle["features"]
print(f"Loaded Random Forest  |  test accuracy {bundle['test_accuracy']:.2%}")

SDSS_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"


def build_feature_row(u, g, r, i, z, redshift):
    """Recreate the exact training features (incl. colour indices) for one object."""
    row = {
        "u": u, "g": g, "r": r, "i": i, "z": z, "redshift": redshift,
        "u_g": u - g, "g_r": g - r, "r_i": r - i, "i_z": i - z,
    }
    return np.array([[row[f] for f in FEATURES]])


@app.route("/")
def index():
    return render_template(
        "index.html",
        accuracy=f"{bundle['test_accuracy']:.2%}",
        f1=f"{bundle['test_f1']:.4f}",
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        vals = [float(data[k]) for k in ("u", "g", "r", "i", "z", "redshift")]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Send numeric values for u, g, r, i, z and redshift."), 400

    X = SCALER.transform(build_feature_row(*vals))
    probs = MODEL.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    return jsonify(
        prediction=CLASSES[idx],
        probabilities={c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        trees_voting=int(round(probs[idx] * MODEL.n_estimators)),
        n_trees=MODEL.n_estimators,
    )


@app.route("/sdss_lookup", methods=["POST"])
def sdss_lookup():
    """Fetch the nearest spectroscopic object to (ra, dec) live from SDSS DR17."""
    data = request.get_json(silent=True) or {}
    try:
        ra, dec = float(data["ra"]), float(data["dec"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Send numeric ra and dec (degrees)."), 400

    sql = (
        "SELECT TOP 1 p.u, p.g, p.r, p.i, p.z, s.z AS redshift, s.class "
        "FROM PhotoObj AS p "
        "JOIN SpecObj AS s ON s.bestobjid = p.objid "
        f"JOIN dbo.fGetNearbySpecObjEq({ra}, {dec}, 3.0) AS n "
        "ON n.specobjid = s.specobjid ORDER BY n.distance"
    )
    try:
        resp = requests.get(SDSS_URL, params={"cmd": sql, "format": "json"}, timeout=15)
        resp.raise_for_status()
        rows = resp.json()[0].get("Rows", [])
    except requests.RequestException:
        return jsonify(error="Could not reach the SDSS SkyServer. Check your "
                             "internet connection and try again."), 502
    except (ValueError, IndexError, KeyError):
        return jsonify(error="Unexpected response from the SDSS SkyServer."), 502

    if not rows:
        return jsonify(error="No spectroscopic SDSS object within 3 arcminutes "
                             "of that position. Try different coordinates."), 404

    o = rows[0]
    return jsonify(
        u=o["u"], g=o["g"], r=o["r"], i=o["i"], z=o["z"],
        redshift=o["redshift"],
        sdss_class=str(o["class"]).strip().upper(),
    )


if __name__ == "__main__":
    # Local development server. In production (Render/railway/etc.) the app
    # is run by gunicorn instead:  gunicorn app:app
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
