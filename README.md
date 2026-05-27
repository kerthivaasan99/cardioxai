<div align="center">

# 🫀 CardioXAI

### An Explainable Stacked Ensemble Framework for Cardiovascular Disease Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.10--3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-6C3483?style=for-the-badge)](https://shap.readthedocs.io/)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://cardioxai-9iq1.onrender.com)
[![License](https://img.shields.io/badge/License-Academic_Use-blue?style=for-the-badge)](#-license)

<br/>

> **CardioXAI** is a full-stack healthcare AI application that predicts cardiovascular disease risk using an XGBoost-based machine learning model and provides transparent, clinically meaningful explanations using **SHAP**, **LIME**, and **DiCE-ML Counterfactual Analysis**.
>
> Designed as a clinical decision support tool for early identification of cardiovascular risk factors and generation of personalised risk-reduction recommendations.

<br/>

🌐 **[Live Demo](https://cardioxai-9iq1.onrender.com)** &nbsp;|&nbsp; 💻 **[GitHub Repository](https://github.com/kerthivaasan99/cardioxai)**

---

</div>

## 📋 Table of Contents

- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Input & Output Features](#-input-features)
- [Installation & Local Setup](#-installation-and-local-setup)
- [Cloud Deployment](#-cloud-deployment)
- [API Endpoints](#-api-endpoints)
- [Application Modules](#-application-modules)
- [Academic Significance](#-academic-significance)
- [Medical Disclaimer](#-medical-disclaimer)
- [License](#-license)

---

## 🎯 Objectives

- Predict cardiovascular disease risk using a stacked ensemble machine learning model
- Provide explainable AI insights through SHAP global and local attributions
- Generate LIME local explanations as an alternative interpretability view
- Produce counterfactual recommendations via DiCE-ML for actionable risk reduction
- Enable real-time What-If scenario simulation for clinical exploration
- Generate downloadable, structured PDF reports for physician review
- Deploy as a cloud-hosted, production-ready web application

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **XGBoost Prediction** | High-performance gradient-boosted ensemble model for risk classification |
| 📊 **SHAP Analysis** | Global feature importance and local per-prediction attribution plots |
| 🔍 **LIME Explanations** | Model-agnostic local interpretable explanations for each prediction |
| 🎯 **Counterfactuals** | DiCE-ML powered "what-if" recommendations to reduce risk class |
| 🎛️ **What-If Simulator** | Real-time scenario analysis by adjusting patient parameters |
| 📄 **PDF Reports** | Automated, structured clinical reports via ReportLab |
| 🌐 **Responsive UI** | Modern web interface compatible with all major browsers |
| ☁️ **Cloud Deployment** | Hosted on Render with automated build via `render.yaml` |

---

## 🛠️ Technology Stack

<table>
<tr>
<td valign="top" width="50%">

### 🤖 Machine Learning
- Python 3.10 – 3.14
- XGBoost
- Scikit-learn
- Pandas
- NumPy

### 🔬 Explainable AI
- SHAP
- LIME
- DiCE-ML

### ⚙️ Backend
- Flask 3.x
- Flask-CORS
- Gunicorn

</td>
<td valign="top" width="50%">

### 🎨 Frontend
- HTML5 / CSS3 / JavaScript
- Chart.js
- Plotly.js

### 📄 Reporting
- ReportLab
- html2pdf.js

### 🚀 Deployment
- GitHub
- Render

</td>
</tr>
</table>

---

## 📂 Project Structure

```text
cardioxai/
├── backend/
│   ├── app.py                    # Flask application entry point
│   ├── explainers.py             # SHAP, LIME, DiCE explainability module
│   ├── preprocessor.py           # Feature preprocessing pipeline
│   ├── requirements.txt          # Python dependency manifest
│   ├── model/
│   │   ├── xgb_model.pkl         # Serialised XGBoost ensemble model
│   │   ├── scaler.pkl            # Fitted StandardScaler artefact
│   │   └── train_model.py        # Model training script
│   └── templates/
│       ├── report_template.html  # PDF report HTML template
│       └── report_styles.css     # Report stylesheet
│
├── frontend/
│   ├── index.html                # Main dashboard UI
│   ├── style.css                 # Global stylesheet
│   └── app.js                    # Frontend logic and API calls
│
├── render.yaml                   # Render cloud deployment config
└── README.md                     # Project documentation
```

---

## 📊 Input Features

| # | Feature | Description |
|---|---|---|
| 1 | **Age** | Patient age in years |
| 2 | **Gender** | Biological sex |
| 3 | **Height** | Height in cm |
| 4 | **Weight** | Weight in kg |
| 5 | **Systolic BP** | Systolic blood pressure (mmHg) |
| 6 | **Diastolic BP** | Diastolic blood pressure (mmHg) |
| 7 | **Cholesterol** | Cholesterol level (1 = Normal, 2 = Above Normal, 3 = High) |
| 8 | **Glucose** | Glucose level (1 = Normal, 2 = Above Normal, 3 = High) |
| 9 | **Smoking** | Smoking status (Yes / No) |
| 10 | **Alcohol** | Alcohol consumption (Yes / No) |
| 11 | **Physical Activity** | Regular physical activity (Yes / No) |

## 📈 Output Features

| Output | Description |
|---|---|
| **Risk Probability** | Cardiovascular disease risk score (0 – 100%) |
| **Risk Category** | Low / Moderate / High classification |
| **Health Score** | Composite wellness score |
| **BMI** | Calculated Body Mass Index |
| **SHAP Explanation** | Feature attribution waterfall / force plot |
| **LIME Explanation** | Local linear approximation per prediction |
| **Counterfactuals** | Actionable parameter changes to reduce risk class |
| **What-If Analysis** | Interactive scenario simulation |
| **PDF Report** | Downloadable physician-ready report |

---

## 🚀 Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vikash1602/cardioxai.git
cd cardioxai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 5. Run the Application

```bash
python backend/app.py
```

### 6. Open in Browser

```
http://127.0.0.1:5000
```

---

## ☁️ Cloud Deployment

The application is deployed on **Render** using the following configuration (`render.yaml`):

```yaml
services:
  - type: web
    name: cardioxai
    env: python
    plan: free
    buildCommand: pip install -r backend/requirements.txt
    startCommand: gunicorn --chdir backend app:app
```

🌐 **Live URL:** [https://cardioxai-9iq1.onrender.com](https://cardioxai-9iq1.onrender.com)

> **Note:** The free-tier Render instance may take 30–60 seconds to wake up if it has been idle. This is expected behaviour.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main dashboard |
| `/health` | `GET` | Service health status check |
| `/predict` | `POST` | Predict cardiovascular disease risk |
| `/counterfactual` | `POST` | Generate personalised recommendations |
| `/generate-report` | `POST` | Generate downloadable PDF report |
| `/debug/explainers` | `GET` | Debug explainability components |

---

## 📷 Application Modules

1. **Patient Data Input Dashboard** — Structured form for entering clinical parameters
2. **Risk Analytics Visualization** — Charts and gauges for risk probability and health score
3. **Explainable AI Analysis** — SHAP and LIME explanation plots
4. **What-If Scenario Simulator** — Real-time parameter adjustment and re-prediction
5. **Counterfactual Recommendations** — Actionable DiCE-ML generated suggestions
6. **PDF Report Generation** — One-click downloadable clinical report

---

## 📚 Academic Significance

CardioXAI demonstrates the practical integration of five engineering disciplines:

| Domain | Contribution |
|---|---|
| **Machine Learning** | Stacked ensemble modelling with XGBoost for clinical prediction |
| **Explainable AI** | SHAP, LIME, and DiCE-ML for transparent, auditable outputs |
| **Full-Stack Development** | Flask REST API + modern HTML/JS/CSS frontend |
| **Healthcare Decision Support** | Risk stratification and personalised clinical recommendations |
| **Cloud Deployment** | Production-grade hosting on Render with CI/CD via Git |

> This project is submitted in partial fulfilment of the requirements for the award of the degree of **Bachelor of Engineering / Bachelor of Technology**, Department of Computer Science and Engineering, **Er. Perumal Manimekalai College of Engineering, Hosur** — Academic Year **2025 – 2026**.

---

## ⚠️ Medical Disclaimer

> This application is intended **for educational and research purposes only**.
> It does **not** provide medical advice and should **not** replace consultation with a qualified healthcare professional.
> All predictions are generated by a machine learning model trained on publicly available datasets and are not clinically validated for real-world diagnostic use.

---

## 📄 License

This project is developed for **academic and research purposes**.
All third-party libraries used are open-source and comply with their respective licences (BSD, MIT, Apache 2.0).

---

