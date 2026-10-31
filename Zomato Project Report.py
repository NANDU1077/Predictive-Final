import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

DATA_FILE = "C:/Users/nandu/OneDrive/Desktop/Predictive Analysis Project/Zomato Restaurant Dataset.csv"
OUTPUT_DIR = "zomato_project_outputs"
TEST_SIZE = 0.20
RANDOM_STATE = 42
DROP_NOT_RATED = True
TOP_N_CUISINES = 20
TOP_N_CITIES = 30
TARGET = "Rating text"
RATING_ORDER = ["Poor", "Average", "Good", "Very Good", "Excellent", "Not rated"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. Put the CSV file in the same folder as this script "
        "or change DATA_FILE to the full path of your dataset."
    )

try:
    data = pd.read_csv(DATA_FILE)
except UnicodeDecodeError:
    data = pd.read_csv(DATA_FILE, encoding="latin-1")
data.columns = data.columns.str.strip()

required_columns = [
    "Restaurant ID", "Restaurant Name", "Country Code", "City", "Longitude", "Latitude",
    "Cuisines", "Average Cost for two", "Currency", "Has Table booking", "Has Online delivery",
    "Is delivering now", "Price range", "Aggregate rating", "Rating color", "Rating text", "Votes"
]
missing_columns = [c for c in required_columns if c not in data.columns]
if missing_columns:
    raise ValueError(
        "Missing expected columns: " + ", ".join(missing_columns) +
        "\nColumns found: " + ", ".join(map(str, data.columns))
    )

print("=" * 60)
print("PREDICTIVE ANALYTICS FOR ZOMATO RESTAURANT RATING CATEGORY")
print("=" * 60)
print("\nDataset shape:", data.shape)
print("\nFirst five rows:\n", data.head())
print("\nDataset information:")
data.info()
print("\nDescriptive statistics:\n", data.describe(include="all").transpose())
print("\nMissing values:\n", data.isnull().sum())
print("\nDuplicate rows:", data.duplicated().sum())
print("\nRating text counts:\n", data[TARGET].value_counts())

data.describe(include="all").transpose().to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"))
data.isnull().sum().rename("missing_values").to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))
data[TARGET].value_counts().rename("count").to_csv(os.path.join(OUTPUT_DIR, "rating_text_counts.csv"))

data = data.drop_duplicates(subset="Restaurant ID").copy()

if DROP_NOT_RATED:
    data = data[data[TARGET] != "Not rated"].copy()

data["Average Cost for two"] = data["Average Cost for two"].replace(0, np.nan)
data["LOG_COST"] = np.log1p(data["Average Cost for two"])
currency_median = data.groupby("Currency")["Average Cost for two"].transform("median")
data["COST_VS_CURRENCY_MEDIAN"] = data["Average Cost for two"] / currency_median
data["LOG_VOTES"] = np.log1p(data["Votes"])

data["Cuisines"] = data["Cuisines"].fillna("Unknown")
data["PRIMARY_CUISINE"] = data["Cuisines"].str.split(",").str[0].str.strip()
top_cuisines = data["PRIMARY_CUISINE"].value_counts().head(TOP_N_CUISINES).index
data["PRIMARY_CUISINE"] = data["PRIMARY_CUISINE"].where(data["PRIMARY_CUISINE"].isin(top_cuisines), "Other")
data["NUM_CUISINES"] = data["Cuisines"].apply(lambda s: 0 if s == "Unknown" else len(s.split(",")))

top_cities = data["City"].value_counts().head(TOP_N_CITIES).index
data["CITY_GROUP"] = data["City"].where(data["City"].isin(top_cities), "Other")
data["COUNTRY_CODE"] = data["Country Code"].astype(str)

for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
    data[col.upper().replace(" ", "_")] = (data[col].str.strip().str.lower() == "yes").astype(int)

leakage_or_identifier_cols = [
    "Aggregate rating", "Rating color", "Restaurant ID", "Restaurant Name", "Address",
    "Locality", "Locality Verbose", "Switch to order menu"
]
print("\nColumns excluded from the model:", leakage_or_identifier_cols)

numeric_columns = [
    "LOG_VOTES", "LOG_COST", "COST_VS_CURRENCY_MEDIAN", "Price range", "NUM_CUISINES",
    "Longitude", "Latitude", "Aggregate rating"
]
eda_features = ["Votes", "Average Cost for two", "Price range", "NUM_CUISINES", "Aggregate rating"]

crop_labels = [r for r in RATING_ORDER if r in data[TARGET].unique()]

plt.figure(figsize=(10, 6))
class_counts = data[TARGET].value_counts().reindex(crop_labels)
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Distribution of Rating Categories")
plt.xlabel("Rating category")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"), dpi=150)
plt.show()
plt.close()

data[numeric_columns].hist(figsize=(14, 10), bins=20)
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "numeric_distributions.png"), dpi=150)
plt.show()
plt.close("all")

corr = data[numeric_columns].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.show()
plt.close(fig)

for feature in eda_features:
    fig, ax = plt.subplots(figsize=(10, 6))
    values = [data.loc[data[TARGET] == r, feature].dropna().values for r in crop_labels]
    ax.boxplot(values, showfliers=True)
    ax.set_xticklabels(crop_labels)
    if feature in ("Votes", "Average Cost for two"):
        ax.set_yscale("log")
    ax.set_title(f"{feature} Across Rating Categories")
    ax.set_xlabel("Rating category")
    ax.set_ylabel(feature)
    fig.tight_layout()
    safe_name = feature.lower().replace(" ", "_")
    fig.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{safe_name}.png"), dpi=150)
    plt.show()
    plt.close(fig)

city_rating = pd.crosstab(data["CITY_GROUP"], data[TARGET])[crop_labels]
city_rating = city_rating.loc[city_rating.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(city_rating.values, aspect="auto", cmap="YlGnBu")
fig.colorbar(im, ax=ax, label="Restaurant count")
ax.set_xticks(range(len(city_rating.columns)))
ax.set_xticklabels(city_rating.columns.astype(str), rotation=45, ha="right")
ax.set_yticks(range(len(city_rating.index)))
ax.set_yticklabels(city_rating.index.astype(str), fontsize=8)
ax.set_title("Rating Categories Across Cities")
ax.set_xlabel("Rating category")
ax.set_ylabel("City")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "city_rating_heatmap.png"), dpi=150)
plt.show()
plt.close(fig)
city_rating.to_csv(os.path.join(OUTPUT_DIR, "city_rating_counts.csv"))

numeric_features = [
    "LOG_VOTES", "LOG_COST", "COST_VS_CURRENCY_MEDIAN", "Price range", "NUM_CUISINES",
    "Longitude", "Latitude", "HAS_TABLE_BOOKING", "HAS_ONLINE_DELIVERY", "IS_DELIVERING_NOW"
]
categorical_features = ["COUNTRY_CODE", "CITY_GROUP", "Currency", "PRIMARY_CUISINE"]

X = data[numeric_features + categorical_features].copy()
y = data[TARGET].copy()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

stratify_target = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_target
)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
}

results = []
labels = crop_labels

for model_name, classifier in models.items():
    print("\n" + "=" * 60 + "\n" + model_name + "\n" + "=" * 60)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1 Score": f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    file_stub = model_name.lower().replace(" ", "_")
    pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    ).to_csv(os.path.join(OUTPUT_DIR, f"{file_stub}_confusion_matrix.csv"))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Number of records")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Rating Category")
    ax.set_ylabel("Actual Rating Category")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(label) for label in labels])
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{file_stub}_confusion_matrix.png"), dpi=150)
    plt.show()
    plt.close(fig)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print("\n" + "=" * 60 + "\nFINAL MODEL COMPARISON\n" + "=" * 60)
print(results_df.round(4).to_string(index=False))
print("\nOutput files saved in:", os.path.abspath(OUTPUT_DIR))