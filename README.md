# GridSense: Automated Multi-Region Energy Forecasting Pipeline

**GridSense** is an industrial-grade MLOps project designed to predict hourly energy demand across multiple power grids. It features an automated **Champion vs. Challenger** model selection pipeline, containerized API serving, and a full CI/CD lifecycle via GitHub Actions.

## 🚀 Project Overview
Energy forecasting is critical for grid stability. GridSense automates the entire process:
1.  **Data Ingestion:** Seamlessly handles 14+ datasets in mixed formats (`.csv` and `.parquet`).
2.  **Automated Training:** Trains multiple models (XGBoost, RandomForest, Ridge) simultaneously.
3.  **Champion Selection:** Automatically selects the model with the lowest Mean Absolute Error (MAE).
4.  **CI/CD Pipeline:** Every code push triggers a fresh training run, unit testing, and Docker build.



## 🏗️ Technical Architecture
The project follows a modular software engineering pattern:

* **`src/preprocess.py`**: Handles mixed-file loading, temporal feature engineering (hour, day, month), and region encoding.
* **`src/train.py`**: The orchestration script that trains "Challenger" models and crowns a "Champion."
* **`api/main.py`**: A high-performance FastAPI backend that serves live predictions.
* **`.github/workflows/ci.yml`**: The "brain" of the project that automates the DevOps lifecycle.

## 📊 Model Performance (Latest Run)
Based on the automated cloud training run, the results were:

| Model | Algorithm | Status |
| :--- | :--- | :--- |
| **RandomForest** | **Ensemble Tree** | **🏆 Champion (MAE: 848.53 MW)** |
| XGBoost | Gradient Boosting | Challenger |
| Ridge | Linear Regression | Challenger |

## 🛠️ Tech Stack
* **Language:** Python 3.9
* **ML Libraries:** Scikit-Learn, XGBoost, Pandas, PyArrow
* **API Framework:** FastAPI + Uvicorn
* **DevOps/MLOps:** Docker, GitHub Actions, Pytest
* **Data Formats:** CSV, Parquet

## 🏃 Getting Started

### 1. Local Development
```powershell
# Install dependencies
pip install -r requirements.txt

# Train models locally
python -m src.train

# Run the API
uvicorn api.main:app --reload
```

### 2. Docker Deployment
```powershell
# Build the image
docker build -t gridsense-api .

# Run the container
docker run -p 8000:8000 gridsense-api
```

## 🧪 Testing
We use `pytest` to ensure the API remains functional during updates. The CI/CD pipeline refuses to build the Docker image if tests fail.
```powershell
python -m pytest tests/
```

---
