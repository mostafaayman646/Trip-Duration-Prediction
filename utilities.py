import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile
from io import BytesIO
import yaml
import os

def evaluate(model, data, features,data_name):
    from sklearn.metrics import r2_score,root_mean_squared_error
    
    y_train_pred = model.predict(data[features])
    
    r2 = r2_score(data.log_trip_duration,y_train_pred)
    RMSE = root_mean_squared_error(data.log_trip_duration,y_train_pred)
    
    print(f"{data_name} has R2 score: {r2} and RMSE: {RMSE}")

def load_model(file_path):
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        print("Error: The file could not be loaded.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_data_from_zip(zip_path,test_size,Shuffle):
    with zipfile.ZipFile(zip_path, 'r') as archive:
        train_df = pd.read_csv(BytesIO(archive.read('split/train.csv')))
        val_df = pd.read_csv(BytesIO(archive.read('split/val.csv')))
        test_df = pd.read_csv(BytesIO(archive.read('split/test.csv')))

    # Combine them all
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Split into train and test
    train, test = train_test_split(
        df,
        test_size=test_size,
        shuffle=Shuffle,
        random_state=17
    )

    return train, test

def save_or_update_yaml(data: dict, filepath: str):
    """
    Save config dict to YAML. If file exists, update it.
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing = yaml.safe_load(f) or {}
        existing.update(data)   # update keys
        data = existing

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Config updated: {filepath}")

def save_configs_and_model(model, numeric_features, categorical_features, args):
    # ----- MODEL CONFIG -----
    model_config = {
        "model_type": "RidgeRegression",
        "model_path": args.model_path,
        "Polynomial_Degree": args.Degree,
        "Alpha": args.Alpha,
        "Scaler": args.preprocessing,
        "Numerical_Features": numeric_features,
        "Categorical_Features": categorical_features
    }
    save_or_update_yaml(model_config, "config/model_config.yaml")

    # ----- DATA CONFIG -----
    data_config = {
        "data_path": args.data_path,
        "train_size": 1-args.test_size,
        "test_size": args.test_size,
        "Shuffle": args.Shuffle,
        "Outlier_Removal": args.outlier,
        "Weak_Features_Drop": args.weak_features_drop
    }
    save_or_update_yaml(data_config, "config/data_config.yaml")

    # ----- MODEL ONLY -----
    joblib.dump(model, "Models/Ridge_Model.pkl")
    print("✅ Model saved as Ridge_Model.pkl")