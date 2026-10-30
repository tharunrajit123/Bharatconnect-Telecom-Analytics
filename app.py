
from pathlib import Path
import sqlite3
import json
import csv
from io import StringIO
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, Response

app = Flask(__name__)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

CHURN_MODEL_PATH = BASE_DIR / "churn_model.pkl"
SWITCH_MODEL_PATH = BASE_DIR / "switch_model.pkl"
DATA_PATH = BASE_DIR / "bharatconnect_sim_switching.csv"
DATABASE = BASE_DIR / "prediction_history.db"


# ==================================================
# 2. SQLITE DATABASE
# ==================================================

def get_db_connection():
    conn = sqlite3.connect(str(DATABASE))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                prediction_type TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL,
                risk_level TEXT,
                details TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_prediction(
    customer_id,
    prediction_type,
    prediction,
    probability,
    risk_level,
    details=None
):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO prediction_history (
                customer_id,
                prediction_type,
                prediction,
                probability,
                risk_level,
                details,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(customer_id or "Guest"),
            str(prediction_type),
            str(prediction),
            float(probability) if probability is not None else None,
            str(risk_level or "Unknown"),
            json.dumps(details or {}, default=str),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


init_database()


# ==================================================
# 3. LOAD MODELS AND DATASET
# ==================================================

churn_model = None
switch_model = None
df = pd.DataFrame()

try:
    if CHURN_MODEL_PATH.exists():
        churn_model = joblib.load(CHURN_MODEL_PATH)
        print("Churn model loaded successfully.")
    else:
        print("WARNING: churn_model.pkl not found.")
except Exception as e:
    print("Churn model loading error:", e)


try:
    if SWITCH_MODEL_PATH.exists():
        switch_model = joblib.load(SWITCH_MODEL_PATH)
        print("SIM switching model loaded successfully.")
    else:
        print("WARNING: switch_model.pkl not found.")
except Exception as e:
    print("SIM switching model loading error:", e)


try:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print("Dataset loaded:", DATA_PATH.name)
        print("Dataset shape:", df.shape)
    else:
        print("WARNING: bharatconnect_sim_switching.csv not found.")
except Exception as e:
    print("Dataset loading error:", e)


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_risk_level(probability):
    if probability <= 30:
        return "Low"
    elif probability <= 60:
        return "Medium"
    return "High"


def get_form_data():
    return {
        "Age": safe_int(request.form.get("age"), 30),
        "Gender": request.form.get("gender", "Male"),
        "State": request.form.get("state", "Tamil Nadu"),
        "Region": request.form.get("region", "Urban"),
        "CurrentSIM": request.form.get("current_sim", "Airtel"),
        "PlanType": request.form.get("plan_type", "Prepaid"),
        "ContractType": request.form.get("contract_type", "No Contract"),
        "TenureMonths": safe_int(request.form.get("tenure"), 12),
        "MonthlyRecharge": safe_float(
            request.form.get("monthly_recharge"), 399
        ),
        "OutstandingBalance": safe_float(
            request.form.get("outstanding_balance"), 0
        ),
        "DaysSinceLastRecharge": safe_int(
            request.form.get("days_since_recharge"), 5
        ),
        "DataUsageGB": safe_float(request.form.get("data_usage"), 10),
        "CallMinutesUsed": safe_int(
            request.form.get("call_minutes"), 300
        ),
        "SMSUsed": safe_int(request.form.get("sms_used"), 50),
        "NetworkType": request.form.get("network_type", "4G"),
        "Device5GCapable": request.form.get("device_5g", "Yes"),
        "5GAvailableArea": request.form.get("5g_available", "Yes"),
        "NetworkSatisfaction": safe_int(
            request.form.get("network_satisfaction"), 3
        ),
        "CallQualityScore": safe_int(
            request.form.get("call_quality"), 3
        ),
        "SupportCalls": safe_int(
            request.form.get("support_calls"), 1
        ),
        "ComplaintsRaised": safe_int(
            request.form.get("complaints"), 0
        ),
        "PlanChangesLastYear": safe_int(
            request.form.get("plan_changes"), 1
        ),
        "PaymentMethod": request.form.get("payment_method", "UPI"),
    }


def prepare_model_input(model, customer_data):
    """
    Keep input feature names and order consistent with
    the features used to train the model.
    """
    feature_columns = getattr(model, "feature_names_in_", None)

    if feature_columns is None:
        raise ValueError(
            "Model training columns are unavailable. "
            "Check the training script and saved model."
        )

    input_df = pd.DataFrame([customer_data])

    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    return input_df[list(feature_columns)]


# ==================================================
# 5. HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# 6. CUSTOMER CHURN PREDICTION
# ==================================================

@app.route("/customer", methods=["GET", "POST"])
def customer():

    if request.method == "GET":
        return render_template("customer.html")

    if churn_model is None:
        return (
            "Churn model not loaded. Check churn_model.pkl.",
            500
        )

    customer_data = get_form_data()
    customer_id = request.form.get("customer_id", "Guest")

    try:
        input_df = prepare_model_input(churn_model, customer_data)

        prediction = churn_model.predict(input_df)[0]

        probability = 0.0

        if hasattr(churn_model, "predict_proba"):
            probabilities = churn_model.predict_proba(input_df)[0]
            classes = list(churn_model.classes_)

            yes_indices = [
                i for i, label in enumerate(classes)
                if str(label).strip().lower()
                in ("yes", "1", "true")
            ]

            if yes_indices:
                probability = float(probabilities[yes_indices[0]]) * 100
            else:
                probability = float(max(probabilities)) * 100
        else:
            if str(prediction).strip().lower() in (
                "yes", "1", "true"
            ):
                probability = 100.0

        churn_label = (
            "Likely to Churn"
            if str(prediction).strip().lower()
            in ("yes", "1", "true")
            else "Likely to Stay"
        )

        risk_level = get_risk_level(probability)

        # Save the churn prediction to SQLite.
        save_prediction(
            customer_id=customer_id,
            prediction_type="Churn Prediction",
            prediction=churn_label,
            probability=probability,
            risk_level=risk_level,
            details=customer_data
        )

        return render_template(
            "result.html",
            customer_data=customer_data,
            prediction=prediction,
            churn_prediction=churn_label,
            churn_probability=round(probability, 2),
            probability=round(probability, 2),
            risk_level=risk_level,
            risk=risk_level
        )

    except Exception as e:
        app.logger.exception("Churn prediction failed")
        return f"Churn prediction error: {e}", 500


# ==================================================
# 7. SIM SWITCHING PREDICTION
# ==================================================

@app.route("/sim-switch", methods=["GET", "POST"])
def sim_switch():

    if request.method == "GET":
        return render_template("switch.html")

    if switch_model is None:
        return (
            "SIM switching model not loaded. "
            "Check switch_model.pkl.",
            500
        )

    customer_data = get_form_data()
    customer_id = request.form.get("customer_id", "Guest")

    try:
        input_df = prepare_model_input(switch_model, customer_data)

        # This model predicts the destination SIM.
        # It does not independently predict whether a switch occurs.
        predicted_sim = str(switch_model.predict(input_df)[0])

        confidence = None

        if hasattr(switch_model, "predict_proba"):
            probabilities = switch_model.predict_proba(input_df)[0]
            confidence = float(max(probabilities)) * 100

        # This is destination prediction confidence, not
        # the probability that the customer will switch.
        save_prediction(
            customer_id=customer_id,
            prediction_type="SIM Switching",
            prediction=predicted_sim,
            probability=confidence,
            risk_level="Unknown",
            details=customer_data
        )

        return render_template(
            "switch_result.html",
            customer_data=customer_data,
            predicted_sim=predicted_sim,
            prediction=predicted_sim,
            recommended_sim=predicted_sim,
            confidence=round(confidence, 2)
            if confidence is not None else None,
            switch_probability=round(confidence, 2)
            if confidence is not None else None
        )

    except Exception as e:
        app.logger.exception("SIM switching prediction failed")
        return f"SIM switching prediction error: {e}", 500


# ==================================================
# 8. ANALYTICS DASHBOARD
# ==================================================

@app.route("/dashboard")
def dashboard():

    data = df.copy()

    sim_counts = {}
    churn_by_sim = {}
    switch_counts = {}
    switch_reasons = {}
    risk_counts = {}
    churn_counts = {}
    plan_counts = {}
    network_counts = {}
    region_counts = {}

    total_customers = int(len(data))
    churned_count = 0
    retained_count = 0
    churn_rate = 0.0
    switchers_count = 0

    most_popular_sim = "N/A"
    highest_churn_sim = "N/A"
    highest_churn_rate = 0.0

    # Customers by SIM
    if "CurrentSIM" in data.columns:
        counts = data["CurrentSIM"].fillna("Unknown").value_counts()
        sim_counts = {str(k): int(v) for k, v in counts.items()}

        if sim_counts:
            most_popular_sim = max(sim_counts, key=sim_counts.get)

    # Churn totals
    if "Churn" in data.columns:
        churn_series = (
            data["Churn"].fillna("Unknown").astype(str).str.strip()
        )

        churn_counts = {
            str(k): int(v)
            for k, v in churn_series.value_counts().items()
        }

        churned_count = int(
            churn_series.str.lower().isin(
                ["yes", "1", "true"]
            ).sum()
        )

        retained_count = max(total_customers - churned_count, 0)

        if total_customers > 0:
            churn_rate = round(
                churned_count / total_customers * 100, 2
            )

    # Churn rate by SIM
    if {"CurrentSIM", "Churn"}.issubset(data.columns):
        for sim in data["CurrentSIM"].dropna().unique():
            sim_data = data[data["CurrentSIM"] == sim]

            labels = (
                sim_data["Churn"].fillna("").astype(str)
                .str.strip().str.lower()
            )

            count = len(sim_data)
            churned = labels.isin(["yes", "1", "true"]).sum()

            churn_by_sim[str(sim)] = round(
                churned / count * 100, 2
            ) if count else 0.0

        if churn_by_sim:
            highest_churn_sim = max(
                churn_by_sim, key=churn_by_sim.get
            )
            highest_churn_rate = churn_by_sim[highest_churn_sim]

    # Switch destination counts
    if "SwitchedSIM" in data.columns:
        switched = (
            data["SwitchedSIM"].fillna("No Switch")
            .astype(str).str.strip()
        )

        actual_switchers = switched[
            switched.str.lower() != "no switch"
        ]

        switch_counts = {
            str(k): int(v)
            for k, v in actual_switchers.value_counts().items()
        }

        switchers_count = int(len(actual_switchers))

    # Switch reasons
    if "SwitchReason" in data.columns:
        if "SwitchedSIM" in data.columns:
            mask = (
                data["SwitchedSIM"].fillna("No Switch")
                .astype(str).str.strip().str.lower()
                .ne("no switch")
            )
            reason_data = data.loc[mask, "SwitchReason"]
        else:
            reason_data = data["SwitchReason"]

        for reason_text in reason_data.dropna().astype(str):
            for reason in reason_text.split(";"):
                reason = reason.strip()

                if reason and reason.lower() not in ("nan", "none"):
                    switch_reasons[reason] = (
                        switch_reasons.get(reason, 0) + 1
                    )

        switch_reasons = dict(
            sorted(
                switch_reasons.items(),
                key=lambda item: item[1],
                reverse=True
            )[:10]
        )

    # Risk levels
    if "RiskLevel" in data.columns:
        counts = data["RiskLevel"].fillna("Unknown").value_counts()
        risk_counts = {str(k): int(v) for k, v in counts.items()}

    # Plan types
    if "PlanType" in data.columns:
        counts = data["PlanType"].fillna("Unknown").value_counts()
        plan_counts = {str(k): int(v) for k, v in counts.items()}

    # Network types
    if "NetworkType" in data.columns:
        counts = data["NetworkType"].fillna("Unknown").value_counts()
        network_counts = {str(k): int(v) for k, v in counts.items()}

    # Regions
    if "Region" in data.columns:
        counts = data["Region"].fillna("Unknown").value_counts()
        region_counts = {str(k): int(v) for k, v in counts.items()}

    return render_template(
        "dashboard.html",
        df=data,
        total_customers=total_customers,
        customers_count=total_customers,
        churned_count=churned_count,
        customers_churned=churned_count,
        churn_rate=churn_rate,
        switchers_count=switchers_count,
        recorded_switchers=switchers_count,
        most_popular_sim=most_popular_sim,
        highest_churn_sim=highest_churn_sim,
        highest_churn_rate=highest_churn_rate,
        sim_counts=sim_counts,
        churn_by_sim=churn_by_sim,
        switch_counts=switch_counts,
        switch_reasons=switch_reasons,
        risk_counts=risk_counts,
        churn_counts=churn_counts,
        retained_count=retained_count,
        plan_counts=plan_counts,
        network_counts=network_counts,
        region_counts=region_counts
    )


# ==================================================
# 9. PREDICTION HISTORY
# ==================================================

@app.route("/history")
def history():

    with get_db_connection() as conn:
        records = conn.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

    return render_template("history.html", records=records)


# ==================================================
# 10. DOWNLOAD HISTORY AS CSV
# ==================================================

@app.route("/history/download")
def download_history():

    with get_db_connection() as conn:
        records = conn.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

    output = StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ID",
        "Customer ID",
        "Prediction Type",
        "Prediction",
        "Probability (%)",
        "Risk Level",
        "Details",
        "Created At"
    ])

    for record in records:
        writer.writerow([
            record["id"],
            record["customer_id"],
            record["prediction_type"],
            record["prediction"],
            record["probability"],
            record["risk_level"],
            record["details"],
            record["created_at"]
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=bharatconnect_prediction_history.csv"
        }
    )


# ==================================================
# 11. RUN APPLICATION
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)
