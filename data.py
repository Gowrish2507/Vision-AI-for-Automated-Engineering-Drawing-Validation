import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
NUM_DRAWINGS = 150          # unique drawing IDs
ROWS_PER_DRAWING = 6        # component types per drawing
COMPONENT_TYPES = ["Wall", "Door", "Window", "Column", "Floor Area", "Room"]

OUTPUT_FILE = "civil_estimation_dataset.csv"


# ---------------------------------------------------------
# HELPER: generate realistic component values
# ---------------------------------------------------------
def generate_component_row(drawing_id, component):
    """Return a dict of realistic values for one component."""

    # --- base realistic ranges per component ---
    if component == "Wall":
        length = np.random.uniform(20, 120)     # ft
        width  = np.random.uniform(0.5, 1.0)    # ft (thickness)
        height = np.random.uniform(9, 12)       # ft
        qty    = np.random.randint(4, 20)

    elif component == "Door":
        length = np.random.uniform(3, 4)        # ft
        width  = np.random.uniform(6.5, 8)      # ft
        height = np.random.uniform(6.5, 7.5)    # ft
        qty    = np.random.randint(2, 10)

    elif component == "Window":
        length = np.random.uniform(3, 6)
        width  = np.random.uniform(3, 5)
        height = np.random.uniform(3, 5)
        qty    = np.random.randint(3, 12)

    elif component == "Column":
        length = np.random.uniform(0.75, 1.5)   # ft
        width  = np.random.uniform(0.75, 1.5)
        height = np.random.uniform(9, 12)
        qty    = np.random.randint(4, 16)

    elif component == "Floor Area":
        length = np.random.uniform(20, 60)
        width  = np.random.uniform(20, 60)
        height = 0
        qty    = 1

    else:  # Room
        length = np.random.uniform(8, 20)
        width  = np.random.uniform(8, 20)
        height = np.random.uniform(9, 12)
        qty    = np.random.randint(1, 6)

    # --- computed area ---
    area = round(length * width, 2)

    # --- drawing-derived values (from OpenCV) with small noise ---
    detected_length = round(length * np.random.uniform(0.96, 1.04), 2)
    detected_area   = round(area   * np.random.uniform(0.96, 1.04), 2)
    detected_qty    = max(1, int(qty + np.random.choice([-1, 0, 1])))

    # --- user estimation (may differ from drawing) ---
    est_length = round(length * np.random.uniform(0.85, 1.15), 2)
    est_area   = round(area   * np.random.uniform(0.85, 1.15), 2)
    est_qty    = max(1, int(qty + np.random.choice([-2, -1, 0, 1, 2])))

    # --- actual / ground truth (used as training target) ---
    actual_length = round(length * np.random.uniform(0.99, 1.01), 2)
    actual_area   = round(area   * np.random.uniform(0.99, 1.01), 2)
    actual_qty    = qty

    # --- cost estimation (rough) ---
    unit_cost = {
        "Wall": 450, "Door": 3500, "Window": 2500,
        "Column": 5500, "Floor Area": 850, "Room": 15000
    }[component]

    estimated_cost = round(est_qty * unit_cost * np.random.uniform(0.9, 1.1), 2)
    actual_cost    = round(actual_qty * unit_cost, 2)

    # --- floor area of drawing (context feature) ---
    floor_area = round(np.random.uniform(600, 2500), 2)

    # --- deviation % (drawing vs estimation) ---
    if detected_area > 0:
        deviation_pct = abs(detected_area - est_area) / detected_area * 100
    else:
        deviation_pct = 0.0

    # --- tolerance (user-accepted) ---
    tolerance = np.random.choice([5, 10, 15, 20])

    # --- validation status (rule-based label) ---
    if deviation_pct <= tolerance:
        status = "Valid"
    elif deviation_pct <= 2 * tolerance:
        status = "Minor Difference"
    else:
        status = "Invalid"

    return {
        "Drawing_ID": drawing_id,
        "Component_Type": component,
        "Length": round(length, 2),
        "Width": round(width, 2),
        "Height": round(height, 2),
        "Area": area,
        "Quantity": qty,
        "Floor_Area": floor_area,

        # Drawing-derived (OpenCV)
        "Detected_Length": detected_length,
        "Detected_Area": detected_area,
        "Detected_Quantity": detected_qty,

        # User estimation input
        "Estimated_Length": est_length,
        "Estimated_Area": est_area,
        "Estimated_Quantity": est_qty,
        "Estimated_Cost": estimated_cost,

        # Actual / ground truth
        "Actual_Length": actual_length,
        "Actual_Area": actual_area,
        "Actual_Quantity": actual_qty,
        "Actual_Cost": actual_cost,

        # Validation meta
        "Tolerance_Percentage": tolerance,
        "Deviation_Percentage": round(deviation_pct, 2),
        "Validation_Status": status
    }


# ---------------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------------
def build_dataset():
    rows = []
    for i in range(1, NUM_DRAWINGS + 1):
        drawing_id = f"DRW_{i:04d}"
        for comp in COMPONENT_TYPES:
            rows.append(generate_component_row(drawing_id, comp))

    df = pd.DataFrame(rows)

    # Shuffle rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Round float columns
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].round(2)

    return df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Dataset created: {OUTPUT_FILE}")
    print(f"   Total rows      : {len(df)}")
    print(f"   Total drawings  : {df['Drawing_ID'].nunique()}")
    print(f"   Components      : {df['Component_Type'].unique().tolist()}")
    print("\n--- Status distribution ---")
    print(df["Validation_Status"].value_counts())
    print("\n--- Sample rows ---")
    print(df.head())