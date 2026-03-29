import os
import json
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from preprocess import load_all_data, engineer_features
from utils import save_artifact

def main():
    print("🔄 1. Loading data (CSV & Parquet)...")
    df = load_all_data()
    print(f"Total rows loaded: {len(df)}")
    
    print("⚙️ 2. Engineering features...")
    df = engineer_features(df, is_training=True)
    
    features = ['hour', 'dayofweek', 'month', 'Region_Code']
    X = df[features]
    y = df['Demand_MW']
    
    # Chronological Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        "Ridge_Linear": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    metrics = {}
    best_model_name = ""
    best_model = None
    lowest_error = float('inf')
    
    print("🚀 3. Training & Evaluating Models...")
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        error = mean_absolute_error(y_test, preds)
        metrics[name] = {"MAE": error}
        save_artifact(model, f"models/{name}.pkl")
        
        if error < lowest_error:
            lowest_error = error
            best_model_name = name
            best_model = model
            
    print(f"🏆 4. Champion Selected: {best_model_name} (MAE: {lowest_error:.2f} MW)")
    save_artifact(best_model, "models/best_model.pkl")
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()