# main_debug.py
from fastapi import FastAPI
import joblib, traceback

app = FastAPI(title="debug")

MODEL_LOADED = False
LOAD_ERROR = None

try:
    print("DEBUG: Attempting to load model.pkl ...", flush=True)
    bundle = joblib.load("model.pkl")
    print("DEBUG: joblib.load returned. Type:", type(bundle), flush=True)
    if not isinstance(bundle, dict):
        raise RuntimeError("Expected model.pkl to be a dict with keys 'model','scaler','columns'")
    for key in ("model","scaler","columns"):
        if key not in bundle:
            raise RuntimeError(f"model.pkl missing key: {key}")
    MODEL_LOADED = True
    print("DEBUG: model bundle loaded successfully. columns count:", len(bundle["columns"]), flush=True)
except Exception:
    LOAD_ERROR = traceback.format_exc()
    print("DEBUG: Error while loading model.pkl:", flush=True)
    print(LOAD_ERROR, flush=True)

@app.get("/health")
def health():
    return {"model_loaded": MODEL_LOADED, "load_error": (LOAD_ERROR[:3000] if LOAD_ERROR else None)}
