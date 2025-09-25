import pandas as pd
from utilities import*
from data_helper import*


if __name__ == '__main__':
    modeling_pipeline = load_model("Ridge_Model.pkl")
    
    args = modeling_pipeline['args']
    
    if args["dataset"] == 1:
        test = pd.read_csv('split_sample/test.csv')

    elif args["dataset"] == 2:
        zip_path = 'split/test.zip'
        filename_inside_zip = 'test.csv'

        test = read_zip_file(zip_path, filename_inside_zip)
    
    test = prepare_data(test,outlier=args["outlier"],weak_features_drop = args["weak_features_drop"])
    
    
    model = modeling_pipeline['model']
    
    Polynomial_Degree    = modeling_pipeline['Polynomial_Degree']
    Scaler               = modeling_pipeline['Scaler']
    Outlier_Removal      = modeling_pipeline['Outlier_Removal']
    Numerical_features   = modeling_pipeline['Numerical features']
    Categorical_features = modeling_pipeline['Categorical features']
    
    test_features = Numerical_features + Categorical_features
    
    print("Model info:")
    print(f"Polynomial_Degree: {Polynomial_Degree}")
    print(f"Scaler: {Scaler}")
    print(f"Outlier_Removal: {Outlier_Removal}")
    print(f"Numerical_features: {Numerical_features}")
    print(f"Categorical_features: {Categorical_features}")
    
    evaluate(model,test,test_features,'Test')