from utilities import*
from data_helper import*


if __name__ == '__main__':
    # -------- Load Configs --------
    with open("config/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    with open("config/data_config.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    
    #Load Test dataset
    _,test= load_data_from_zip('Data/split.zip',test_size=data_config['test_size'],Shuffle=data_config['Shuffle'])
    test = prepare_data(
        test,
        outlier=data_config["Outlier_Removal"],
        weak_features_drop=data_config["Weak_Features_Drop"]
    )
    #Load model
    model = load_model("Models/Ridge_Model.pkl")
    
    Numerical_features   = model_config["Numerical_Features"]
    Categorical_features = model_config["Categorical_Features"]
    
    test_features = Numerical_features + Categorical_features
    
    evaluate(model,test,test_features,'Test')