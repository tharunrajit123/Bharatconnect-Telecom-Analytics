# BharatConnect – Telecom Customer Intelligence Platform

## Project Overview

BharatConnect is a Data Science and Machine Learning web application designed to analyze telecom customer behavior, predict customer churn, and estimate which SIM provider a customer may switch to.

The platform combines Machine Learning models, data analytics, an interactive dashboard, and prediction history to support telecom customer retention analysis.

## Objectives

- Analyze telecom customer demographics and service usage.
- Predict whether a customer is likely to churn.
- Predict the potential SIM switching destination.
- Analyze customer churn rates across telecom providers.
- Visualize customer insights through an analytics dashboard.
- Store prediction results and download prediction history.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Chart.js
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript
- **Development Environment:** Visual Studio Code

## Key Features

### 1. Customer Churn Prediction
Predicts customer churn using a trained Machine Learning model and displays the prediction probability and risk level.

### 2. SIM Switching Prediction
Predicts a potential SIM switching destination, including Airtel, Jio, Vi, BSNL, or No Switch, based on the trained model.

### 3. Analytics Dashboard
Displays customer statistics, churn rate, current SIM distribution, SIM-wise churn rates, switching destinations, switching reasons, and customer risk distribution.

### 4. Prediction History
Stores prediction records in an SQLite database and provides a history page with CSV download functionality.

### 5. Customer Insights
Helps explore customer retention patterns, network experience, recharge behavior, and switching trends.

## Project Structure

```text
Telecom_web_app/
│
├── app.py
├── train_model.py
├── train_switch_model.py
├── churn_model.pkl
├── switch_model.pkl
├── telecom_cleaned.csv
├── bharatconnect_sim_switching.csv
├── prediction_history.db
│
├── templates/
│   ├── index.html
│   ├── customer.html
│   ├── result.html
│   ├── switch.html
│   ├── switch_result.html
│   ├── dashboard.html
│   └── history.html
│
├── static/
│   └── css/
│       └── style.css
│
└── README.md
```

*Note: `prediction_history.db` may be created automatically when the application runs. The structure above reflects the current project files; adjust it if your folder differs.*

## Dataset

The project uses telecom customer data containing features such as:

- Customer age, gender, state, and region
- Current SIM provider and plan type
- Customer tenure and monthly recharge
- Data usage, call minutes, and SMS usage
- Network type and network satisfaction
- Customer complaints and support calls
- Churn status and SIM switching information
- Switching reasons and risk labels

The dataset supports customer behavior analysis and Machine Learning experiments. Any generated or simulated records should be identified as such when presenting the project results.

## Installation and Setup

### Step 1: Install Python

Install Python and verify the installation:

```bash
python --version
```

### Step 2: Open the project folder

Open the `Telecom_web_app` folder in Visual Studio Code.

### Step 3: Install required libraries

Open the VS Code terminal and run:

```bash
python -m pip install flask pandas numpy scikit-learn joblib
```

### Step 4: Check the project files

Ensure that the required model files and datasets are available in the project folder:

- `churn_model.pkl`
- `switch_model.pkl`
- `bharatconnect_sim_switching.csv`

The churn dataset `telecom_cleaned.csv` should also be present if required by your application.

### Step 5: Run the application

```bash
python app.py
```

### Step 6: Open the website

Open this URL in your browser:

http://127.0.0.1:5000

## Machine Learning Workflow

1. Load the telecom customer dataset.
2. Inspect and prepare the data.
3. Encode categorical variables and prepare model features.
4. Train the churn prediction model.
5. Train the SIM switching prediction model.
6. Evaluate the models using suitable classification metrics.
7. Integrate the trained models into the Flask application.
8. Display predictions and analytics through the website.

## Business Benefits

- Identify customer groups that may need retention support.
- Compare churn patterns across SIM providers.
- Understand recorded switching trends and reasons.
- Explore customer service and network-related factors.
- Present analytical findings through a centralized dashboard.

## Future Enhancements

- Improve model performance and evaluate class-wise prediction metrics.
- Add interactive filters for SIM provider, state, and region.
- Introduce model monitoring and periodic retraining.
- Deploy the application to a cloud platform.
- Add authentication and role-based access.

## Conclusion

BharatConnect demonstrates how Data Science, Machine Learning, web development, and data visualization can be integrated into a telecom customer intelligence application. It provides a practical platform for exploring churn behavior, predicting potential SIM switching, and communicating customer insights through an interactive dashboard.

**Disclaimer:** Predictions are estimates produced by trained models and should not be treated as guaranteed customer behavior. Model performance depends on the quality and representativeness of the training data.
