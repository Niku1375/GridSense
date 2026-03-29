import joblib
import os

def save_artifact(artifact, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(artifact, filepath)
    print(f"Artifact saved to {filepath}")

def load_artifact(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)