
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("bharatconnect_sim_switching.csv")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Remove customers below 18
df = df[df["Age"] >= 18].copy()

# 3. Keep customers who actually switched SIM
df = df[df["SwitchedSIM"] != "No Switch"].copy()

# 4. Remove output-related columns to avoid data leakage
drop_cols = [
    "CustomerID",
    "SwitchedSIM",
    "SwitchProbability",
    "RiskLevel",
    "SwitchReason"
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df["SwitchedSIM"]


# 5. Prepare categorical and numerical columns

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()


# 6. Preprocessing with missing-value handling

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

preprocessor = ColumnTransformer([
    (
        "categorical",
        categorical_pipeline,
        categorical_cols
    ),
    (
        "numerical",
        numerical_pipeline,
        numerical_cols
    )
])

# 7. Build model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Train
print("Training SIM Switching Model...")
model.fit(X_train, y_train)

# 9. Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

# 10. Save model
joblib.dump(model, "switch_model.pkl")

# Save the exact input columns for website integration
joblib.dump(X.columns.tolist(), "switch_model_columns.pkl")

print("\nSwitch model saved successfully!")
print("Files: switch_model.pkl, switch_model_columns.pkl")