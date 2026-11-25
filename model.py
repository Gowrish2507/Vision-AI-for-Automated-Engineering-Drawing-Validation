import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR       = r"C:\Users\CRESCENT DEV\Desktop\ENGDrawing Validation"
DATA_PATH      = os.path.join(BASE_DIR, "dataset", "civil_estimation_dataset.csv")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
REPORT_DIR     = os.path.join(BASE_DIR, "reports", "model_results")
GRAPH_DIR      = os.path.join(BASE_DIR, "graphs", "model_graphs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

REPORT_TXT = os.path.join(REPORT_DIR, "model_evaluation.txt")

# ---------------------------------------------------------
# LOGGER
# ---------------------------------------------------------
class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

logger = Logger(REPORT_TXT)
sys.stdout = logger

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
section("1. LOADING DATASET")
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset with shape: {df.shape}")

# ---------------------------------------------------------
# 2. PREPROCESSING & FEATURE SELECTION
# ---------------------------------------------------------
section("2. PREPROCESSING & FEATURE SELECTION")

# Encode categorical columns
le_component = LabelEncoder()
df["Component_Type_Encoded"] = le_component.fit_transform(df["Component_Type"])

le_status = LabelEncoder()
df["Validation_Status_Encoded"] = le_status.fit_transform(df["Validation_Status"])

# --- Feature List 1: For Regressor (Predicting Quantity) ---
# NO Deviation here, because we are predicting the Quantity, not the Status.
features_reg = [
    "Component_Type_Encoded",
    "Detected_Length",
    "Detected_Area",
    "Detected_Quantity",
    "Estimated_Length",
    "Estimated_Area",
    "Estimated_Quantity",
    "Floor_Area",
    "Tolerance_Percentage"
]

# --- Feature List 2: For Classifier (Predicting Status) ---
# WE ADD Deviation_Percentage here. 
# In your app.py, you will calculate this using OpenCV and user inputs.
features_cls = features_reg + ["Deviation_Percentage"]

X_reg = df[features_reg]
X_cls = df[features_cls]

y_reg = df["Actual_Quantity"]
y_cls = df["Validation_Status_Encoded"]

print(f"Regressor Features ({len(features_reg)})")
print(f"Classifier Features ({len(features_cls)}) -> Includes Deviation %")

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------
section("3. TRAIN / TEST SPLIT")
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

print(f"Train size: {X_reg_train.shape[0]} samples")
print(f"Test size : {X_reg_test.shape[0]} samples")

# ---------------------------------------------------------
# 4. REGRESSION MODEL (Predict Quantity)
# ---------------------------------------------------------
section("4. RANDOM FOREST REGRESSOR")
# Increased depth slightly
reg_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
reg_model.fit(X_reg_train, y_reg_train)

y_reg_pred = reg_model.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Root Mean Squared Error   : {rmse:.4f}")
print(f"R² Score                  : {r2:.4f}")

joblib.dump(reg_model, os.path.join(MODEL_DIR, "rf_regressor.pkl"))

plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.6, color="teal")
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
plt.title("Regressor: Actual vs Predicted Quantity")
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "reg_actual_vs_pred.png"))
plt.close()

plt.figure(figsize=(10, 6))
pd.Series(reg_model.feature_importances_, index=features_reg).sort_values().plot(kind="barh", color="teal")
plt.title("Feature Importance (Random Forest Regressor)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "reg_feature_importance.png"))
plt.close()

# ---------------------------------------------------------
# 5. CLASSIFICATION MODEL (Predict Validation Status)
# ---------------------------------------------------------
section("5. RANDOM FOREST CLASSIFIER")
# Increased depth & estimators for better accuracy
cls_model = RandomForestClassifier(
    n_estimators=200, max_depth=20, random_state=42, class_weight="balanced"
)
cls_model.fit(X_cls_train, y_cls_train)

y_cls_pred = cls_model.predict(X_cls_test)

acc = accuracy_score(y_cls_test, y_cls_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_cls_test, y_cls_pred, target_names=le_status.classes_))

joblib.dump(cls_model, os.path.join(MODEL_DIR, "rf_classifier.pkl"))
joblib.dump(le_component, os.path.join(MODEL_DIR, "le_component.pkl"))
joblib.dump(le_status, os.path.join(MODEL_DIR, "le_status.pkl"))

cm = confusion_matrix(y_cls_test, y_cls_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_status.classes_, yticklabels=le_status.classes_)
plt.title("Confusion Matrix (Validation Status)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "cls_confusion_matrix.png"))
plt.close()

plt.figure(figsize=(10, 6))
pd.Series(cls_model.feature_importances_, index=features_cls).sort_values().plot(kind="barh", color="darkorange")
plt.title("Feature Importance (Random Forest Classifier)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "cls_feature_importance.png"))
plt.close()

# ---------------------------------------------------------
# 6. DONE
# ---------------------------------------------------------
section("✅ MODEL TRAINING COMPLETE")
print(f"📦 Models saved to     : {MODEL_DIR}")
print(f"📄 Report saved to     : {REPORT_TXT}")
print(f"🖼️  Graphs saved to     : {GRAPH_DIR}")

logger.close()
sys.stdout = logger.terminal