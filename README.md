✈️ Predictive Maintenance for Aircraft Turbofan Engines

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Status](https://img.shields.io/badge/Status-Active-green)


--------------------------------------- Presently - System - is - Under Maintainance -------------------------------------------------------

(Sorry for inconvenience - try out after finishing maintainance work)

🎯 Problem
Predict Remaining Useful Life (RUL) of aircraft turbofan engines
before breakdown using real NASA sensor data.

📊 Dataset
NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- 100 engines run to failure
- 21 sensors per engine
- 20,631 training samples

🏗️ Pipeline

Data → EDA → Feature Engineering → ML Models → API → Deployment
Live API

Endpoint    : https://aircraft-predictive-maintenance.onrender.com
API Docs    : https://aircraft-predictive-maintenance.onrender.com/docs
Dashboard   : https://aircraft-predictive-maintenance-asy28vaegwsxdncenrirqw.streamlit.app


Model Performance

Our AI model was tested on 20 aircraft engines it had never seen before.
Results show how accurately it predicts Remaining Useful Life (RUL) —
the number of cycles left before an engine needs maintenance.

| Version | Model                 | Accuracy       | Technical RMSE | Status         |
|---------|-----------------------|----------------|----------------|--------------- |
| v1.0    | Random Forest (tuned) | ±15 cycles     | 15.37 cycles   | ✅ Deployed    |
| v2.0    | LSTM Neural Network   | TBD            | TBD            | 🔄 In Progress |
| v3.0    | Transformer           | TBD            | TBD            | 📅 Planned     |

What does ±15 cycles mean?

If an engine has 100 cycles remaining before maintenance is needed,
our model predicts somewhere between 85 and 115 cycles.
For an engine with an average 200 cycle lifespan, that is 92% accuracy.

Think of it like a fuel gauge in your car — it does not tell you the
exact kilometers remaining, but it reliably warns you when to refuel
before you are stranded. Our model works the same way for aircraft engines,
giving maintenance teams early warning before failure occurs.

Why this matters?

Unplanned engine failure costs airlines $500,000+ per incident.
Early and accurate RUL prediction allows:
  → Scheduled maintenance instead of emergency repairs
  → Reduced downtime and flight cancellations
  → Significant cost savings for airlines
  → Most importantly — safer flights

📁 Project Structure

    aircraft_predictive_maintenance/
    ├── data/           → NASA C-MAPSS dataset
    ├── notebooks/      → EDA and experiments
    ├── models/         → saved model files
    ├── src/            → source code
    ├── api/            → FastAPI application
    └── dashboard/      → monitoring dashboard

⚙️ Setup

    git clone https://github.com/USERNAME/aircraft-predictive-maintenance.git
    cd aircraft-predictive-maintenance

Create conda environment:

    conda create -n pm_env python=3.11
    conda activate pm_env
    pip install -r requirements.txt

📈 Key Findings
- Identified 7 dead sensors, retained 11 high-signal sensors
- Dominant feature: sensor_11 (bypass ratio) confirms engine physics
- RUL capped at 125 cycles using piecewise linear approach
- Minimal overfitting gap of 2.76 cycles on baseline RF

🛠️ Tech Stack
- Language  : Python 3.11 (Anaconda)
- ML Models : scikit-learn, PyTorchc
- API       : FastAPI
- Dashboard : Streamlit
- Deployment: Render / HuggingFace Spaces
