# 🌾 Smart Agriculture Yield Prediction

## 📌 Project Overview

Smart Agriculture Yield Prediction is a Machine Learning and MLOps project that predicts crop yield based on agricultural factors such as crop type, state, district, season, cultivation area, and crop year.

The project uses a Random Forest Regressor model and follows MLOps practices including MLflow Experiment Tracking, Model Registry, GitHub version control, and Streamlit deployment.

---

## 🚀 Features

* Crop Yield Prediction
* Random Forest Regression Model
* Feature Engineering and Data Preprocessing
* MLflow Experiment Tracking
* MLflow Model Registry
* Streamlit Web Application
* GitHub Version Control
* End-to-End MLOps Workflow

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* MLflow
* Streamlit
* Joblib
* GitHub

---

## 📂 Project Structure

```text
Smart-Agriculture-Yield-Prediction/
│
├── app/
│   └── app.py
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── test.py
│   └── utils.py
│
├── models/
│   ├── crop_yield_model.pkl
│   └── crop_yield_encoders.pkl
│
├── data/
│   └── crop_production.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Model Performance

| Metric      | Value  |
| ----------- | ------ |
| Train Score | 0.9695 |
| Test Score  | 0.9612 |
| R² Score    | 0.9612 |
| MAE         | 0.1185 |

Algorithm Used:

* Random Forest Regressor

---

## 🔄 MLflow Tracking

The project uses MLflow for:

* Experiment Tracking
* Parameter Logging
* Metric Logging
* Model Logging
* Model Registry

Logged Parameters:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf
* random_state

Logged Metrics:

* Train Score
* Test Score
* R² Score
* MAE
* RMSE

---

## 🌐 Streamlit Application

The Streamlit application allows users to:

1. Select Crop
2. Select State
3. Select District
4. Select Season
5. Enter Area
6. Select Crop Year
7. Predict Crop Yield

Output:

* Predicted Yield (tonnes/hectare)
* Estimated Total Production (tonnes)

---

## ▶️ Running Locally

### Clone Repository

```bash
git clone <repository-url>
cd Smart-Agriculture-Yield-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python src/train.py
```

### Launch MLflow

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

### Run Streamlit App

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

## 🎯 Business Use Case

This project helps:

* Farmers
* Agricultural Analysts
* Government Agencies
* Agritech Companies

to estimate crop yield and expected production before harvest.

---

## 👨‍💻 Author

Tushar Sakhamuri

AI Engineer | Machine Learning Engineer | Generative AI Enthusiast

GitHub:
https://github.com/tusharsakhamuri9-hue

LinkedIn:
https://www.linkedin.com/in/tusharsakhamuri/
