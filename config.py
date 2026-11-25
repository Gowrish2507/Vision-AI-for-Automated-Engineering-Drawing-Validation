import os
import joblib
import pytesseract

# Tesseract Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Paths
BASE_DIR = r"C:\Users\CRESCENT DEV\Desktop\ENGDrawing Validation"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load Models Safely
def load_ml_models():
    try:
        reg = joblib.load(os.path.join(MODEL_DIR, "rf_regressor.pkl"))
        clf = joblib.load(os.path.join(MODEL_DIR, "rf_classifier.pkl"))
        le_comp = joblib.load(os.path.join(MODEL_DIR, "le_component.pkl"))
        le_stat = joblib.load(os.path.join(MODEL_DIR, "le_status.pkl"))
        return reg, clf, le_comp, le_stat
    except Exception as e:
        raise FileNotFoundError(f"Models not found. Run model.py first. Error: {e}")

reg_model, cls_model, le_comp, le_stat = load_ml_models()