import zipfile
import joblib
import pandas as pd

def evaluate(model, data, features,data_name):
    from sklearn.metrics import r2_score
    
    y_train_pred = model.predict(data[features])
    
    r2 = r2_score(data.log_trip_duration,y_train_pred)
    
    print(f"{data_name} has R2 score: {r2}")

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        print("Error: The file could not be loaded.")
    except Exception as e:
        print(f"An error occurred: {e}")

def read_zip_file(zip_path, filename_inside_zip):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # read CSV directly from inside the zip
        with zip_ref.open(filename_inside_zip) as f:
            return pd.read_csv(f)