import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------

df = pd.read_csv("telecom_cleaned.csv")

print("Dataset Shape:", df.shape)


# ------------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------------

features = [
    "Age",
    "Gender",
    "State",
    "CurrentSIM",
    "PlanType",
    "ContractType",
    "TenureMonths",
    "MonthlyRecharge",
    "DataUsageGB",
    "NetworkType",
    "NetworkSatisfaction",
    "CallQualityScore",
    "SupportCalls",
    "ComplaintsRaised",
    "5GAvailableArea"
]


X = df[features].copy()

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ------------------------------------------------
# 3. NUMERICAL FEATURES
# ------------------------------------------------

numeric_features = [
    "Age",
    "TenureMonths",
    "MonthlyRecharge",
    "DataUsageGB",
    "NetworkSatisfaction",
    "CallQualityScore",
    "SupportCalls",
    "ComplaintsRaised"
]


# ------------------------------------------------
# 4. CATEGORICAL FEATURES
# ------------------------------------------------

categorical_features = [
    "Gender",
    "State",
    "CurrentSIM",
    "PlanType",
    "ContractType",
    "NetworkType",
    "5GAvailableArea"
]


# ------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------
# 6. MODEL
# ------------------------------------------------

model = LogisticRegression(
    max_iter=2000
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ------------------------------------------------
# 7. TRAIN TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# 8. TRAIN MODEL
# ------------------------------------------------

print("Training model...")

pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------
# 9. TEST ACCURACY
# ------------------------------------------------

accuracy = pipeline.score(
    X_test,
    y_test
)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)


# ------------------------------------------------
# 10. SAVE MODEL
# ------------------------------------------------

joblib.dump(
    pipeline,
    "churn_model.pkl"
)

print("Model saved successfully!")

print("File created: churn_model.pkl")