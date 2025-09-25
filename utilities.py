import joblib
import pandas as pd

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
