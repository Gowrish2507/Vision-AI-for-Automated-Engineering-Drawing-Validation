import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR   = r"C:\Users\CRESCENT DEV\Desktop\ENGDrawing Validation"
DATA_PATH  = os.path.join(BASE_DIR, "dataset", "civil_estimation_dataset.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
GRAPH_DIR  = os.path.join(BASE_DIR, "graphs")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

REPORT_TXT   = os.path.join(REPORT_DIR, "analysis_report.txt")
SUMMARY_CSV  = os.path.join(REPORT_DIR, "summary_stats.csv")
CORR_CSV     = os.path.join(REPORT_DIR, "correlation_matrix.csv")
STATUS_CSV   = os.path.join(REPORT_DIR, "validation_status_distribution.csv")

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

# ---------------------------------------------------------
# LOGGER (writes to both console & report file)
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
print(f"Dataset path : {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found. Please check the path.")
    logger.close()
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded successfully.")
print(f"Shape        : {df.shape}  (rows, columns)")
print(f"Columns      : {list(df.columns)}")


# ---------------------------------------------------------
# 2. BASIC INFO
# ---------------------------------------------------------
section("2. BASIC INFORMATION")
print("\n--- dtypes ---")
print(df.dtypes)

print("\n--- first 5 rows ---")
print(df.head())

print("\n--- last 5 rows ---")
print(df.tail())


# ---------------------------------------------------------
# 3. MISSING / DUPLICATES
# ---------------------------------------------------------
section("3. MISSING VALUES & DUPLICATES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Percent": missing_pct})
print("\n--- missing values ---")
print(missing_df[missing_df["Missing"] > 0] if missing.sum() > 0 else "None ✅")

dups = df.duplicated().sum()
print(f"\nDuplicate rows : {dups}")


# ---------------------------------------------------------
# 4. SUMMARY STATISTICS
# ---------------------------------------------------------
section("4. SUMMARY STATISTICS")
summary = df.describe(include="all").T
print(summary)
summary.to_csv(SUMMARY_CSV)
print(f"\n✅ Saved summary stats → {SUMMARY_CSV}")


# ---------------------------------------------------------
# 5. CATEGORICAL DISTRIBUTIONS
# ---------------------------------------------------------
section("5. CATEGORICAL DISTRIBUTIONS")

if "Component_Type" in df.columns:
    print("\n--- Component_Type counts ---")
    print(df["Component_Type"].value_counts())

if "Validation_Status" in df.columns:
    print("\n--- Validation_Status counts ---")
    status_counts = df["Validation_Status"].value_counts()
    print(status_counts)
    status_counts.to_csv(STATUS_CSV, header=["count"])
    print(f"✅ Saved status distribution → {STATUS_CSV}")


# ---------------------------------------------------------
# 6. CORRELATION MATRIX
# ---------------------------------------------------------
section("6. CORRELATION MATRIX")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr(numeric_only=True).round(3)
corr.to_csv(CORR_CSV)
print(corr)
print(f"\n✅ Saved correlation matrix → {CORR_CSV}")


# ---------------------------------------------------------
# 7. GRAPHS
# ---------------------------------------------------------
section("7. GENERATING GRAPHS")

# ---------- 01: numeric distributions ----------
numeric_cols = numeric_df.columns.tolist()
n = len(numeric_cols)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(col, fontsize=9)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.suptitle("Numeric Feature Distributions", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "01_numeric_distributions.png"))
plt.close()
print("✅ 01_numeric_distributions.png")

# ---------- 02: Component_Type count ----------
if "Component_Type" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Component_Type",
                  order=df["Component_Type"].value_counts().index,
                  palette="viridis")
    plt.title("Component Type Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "02_component_type_count.png"))
    plt.close()
    print("✅ 02_component_type_count.png")

# ---------- 03: Validation_Status count ----------
if "Validation_Status" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Validation_Status",
                  order=df["Validation_Status"].value_counts().index,
                  palette="Set2")
    plt.title("Validation Status Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "03_validation_status_count.png"))
    plt.close()
    print("✅ 03_validation_status_count.png")

# ---------- 04: correlation heatmap ----------
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "04_correlation_heatmap.png"))
plt.close()
print("✅ 04_correlation_heatmap.png")

# ---------- 05: boxplots by Component ----------
if "Component_Type" in df.columns:
    key_numeric = [c for c in ["Detected_Area", "Estimated_Area",
                               "Actual_Area", "Deviation_Percentage"]
                   if c in df.columns]
    if key_numeric:
        fig, axes = plt.subplots(1, len(key_numeric),
                                 figsize=(5 * len(key_numeric), 5))
        if len(key_numeric) == 1:
            axes = [axes]
        for ax, col in zip(axes, key_numeric):
            sns.boxplot(data=df, x="Component_Type", y=col,
                        ax=ax, palette="Set3")
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=30)
        plt.suptitle("Boxplots by Component Type", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, "05_boxplots_by_component.png"))
        plt.close()
        print("✅ 05_boxplots_by_component.png")

# ---------- 06: Deviation vs Tolerance ----------
if {"Deviation_Percentage", "Tolerance_Percentage"}.issubset(df.columns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Tolerance_Percentage",
                    y="Deviation_Percentage",
                    hue="Validation_Status" if "Validation_Status" in df.columns else None,
                    palette="Set1", alpha=0.7)
    plt.axline((0, 0), slope=1, color="red", linestyle="--",
               label="Deviation = Tolerance")
    plt.title("Deviation % vs Tolerance %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "06_deviation_vs_tolerance.png"))
    plt.close()
    print("✅ 06_deviation_vs_tolerance.png")

# ---------- 07: Area comparison ----------
area_cols = [c for c in ["Detected_Area", "Estimated_Area", "Actual_Area"]
             if c in df.columns]
if area_cols:
    plt.figure(figsize=(10, 6))
    df[area_cols].plot(kind="kde", ax=plt.gca())
    plt.title("Area Comparison (Detected vs Estimated vs Actual)")
    plt.xlabel("Area")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "07_area_comparison.png"))
    plt.close()
    print("✅ 07_area_comparison.png")

# ---------- 08: Quantity comparison ----------
qty_cols = [c for c in ["Detected_Quantity", "Estimated_Quantity",
                        "Actual_Quantity", "Quantity"]
            if c in df.columns]
if qty_cols:
    plt.figure(figsize=(10, 6))
    df[qty_cols].plot(kind="box", ax=plt.gca())
    plt.title("Quantity Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "08_quantity_comparison.png"))
    plt.close()
    print("✅ 08_quantity_comparison.png")

# ---------- 09: Cost comparison ----------
cost_cols = [c for c in ["Estimated_Cost", "Actual_Cost"]
             if c in df.columns]
if cost_cols:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Estimated_Cost", y="Actual_Cost",
                    alpha=0.6, color="darkgreen")
    plt.title("Estimated Cost vs Actual Cost")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "09_cost_comparison.png"))
    plt.close()
    print("✅ 09_cost_comparison.png")

# ---------- 10: Deviation distribution ----------
if "Deviation_Percentage" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Deviation_Percentage"], kde=True, bins=30,
                 color="crimson")
    plt.axvline(df["Deviation_Percentage"].mean(), color="black",
                linestyle="--", label=f"Mean = {df['Deviation_Percentage'].mean():.2f}%")
    plt.title("Deviation Percentage Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "10_deviation_distribution.png"))
    plt.close()
    print("✅ 10_deviation_distribution.png")

# ---------- 11: Floor_Area vs Deviation ----------
if {"Floor_Area", "Deviation_Percentage"}.issubset(df.columns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Floor_Area", y="Deviation_Percentage",
                    hue="Component_Type" if "Component_Type" in df.columns else None,
                    alpha=0.6)
    plt.title("Floor Area vs Deviation %")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "11_floor_area_vs_deviation.png"))
    plt.close()
    print("✅ 11_floor_area_vs_deviation.png")

# ---------- 12: Status by component ----------
if {"Validation_Status", "Component_Type"}.issubset(df.columns):
    ct = pd.crosstab(df["Component_Type"], df["Validation_Status"])
    ct.plot(kind="bar", stacked=True, figsize=(10, 6),
            colormap="Set2")
    plt.title("Validation Status by Component Type")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "12_status_by_component.png"))
    plt.close()
    print("✅ 12_status_by_component.png")

# ---------- 13: pairplot (key features) ----------
pair_cols = [c for c in ["Detected_Area", "Estimated_Area", "Actual_Area",
                         "Deviation_Percentage", "Tolerance_Percentage"]
             if c in df.columns]
if len(pair_cols) >= 3:
    sample_df = df[pair_cols + (["Validation_Status"]
                                if "Validation_Status" in df.columns else [])]
    pp = sns.pairplot(sample_df.dropna(),
                      hue="Validation_Status" if "Validation_Status" in sample_df.columns else None,
                      palette="Set1", corner=True, plot_kws={"s": 12, "alpha": 0.6})
    pp.fig.suptitle("Pairplot of Key Features", y=1.02)
    pp.savefig(os.path.join(GRAPH_DIR, "13_pairplot_key_features.png"))
    plt.close()
    print("✅ 13_pairplot_key_features.png")


# ---------------------------------------------------------
# 8. KEY INSIGHTS
# ---------------------------------------------------------
section("8. KEY INSIGHTS")

insights = []

if "Validation_Status" in df.columns:
    dist = df["Validation_Status"].value_counts(normalize=True) * 100
    for k, v in dist.items():
        insights.append(f"• {k}: {v:.2f}% of records")

if "Deviation_Percentage" in df.columns:
    insights.append(f"• Mean deviation: {df['Deviation_Percentage'].mean():.2f}%")
    insights.append(f"• Max deviation : {df['Deviation_Percentage'].max():.2f}%")
    insights.append(f"• Min deviation : {df['Deviation_Percentage'].min():.2f}%")

if "Component_Type" in df.columns:
    top_comp = df["Component_Type"].value_counts().idxmax()
    insights.append(f"• Most common component: {top_comp}")

if {"Estimated_Cost", "Actual_Cost"}.issubset(df.columns):
    cost_diff = (df["Estimated_Cost"] - df["Actual_Cost"]).mean()
    insights.append(f"• Average cost diff (Est - Act): {cost_diff:,.2f}")

for line in insights:
    print(line)


# ---------------------------------------------------------
# 9. DONE
# ---------------------------------------------------------
section("✅ ANALYSIS COMPLETE")
print(f"📄 Text report     → {REPORT_TXT}")
print(f"📄 Summary stats   → {SUMMARY_CSV}")
print(f"📄 Correlation     → {CORR_CSV}")
print(f"📄 Status dist.    → {STATUS_CSV}")
print(f"🖼️  Graphs folder   → {GRAPH_DIR}")
print(f"🖼️  Total graphs    → {len(os.listdir(GRAPH_DIR))}")

logger.close()
sys.stdout = logger.terminal