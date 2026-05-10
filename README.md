✈️ Predictive Maintenance for Aircraft Turbofan Engines

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

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

🤖 Model Evolution

| Version | Model                   | Val RMSE     | Status         |
|---------|-------------------------|--------------|----------------|
| v1.0    | Random Forest (tuned)   | 15.37 cycles | ✅ Deployed    |
| v2.0    | LSTM                    | TBD          | 🔄 In Progress |
| v3.0    | Transformer             | TBD          | 📅 Planned     |

🚀 API
Live endpoint: coming soon

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