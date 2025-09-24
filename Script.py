import argparse
import pandas as pd
import os
from utilities import*
from data_helper import*

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=2,
                    help =  '1 for split_sample'
                            '2 for full dataset')

parser.add_argument('--outlier', type=bool, default=False)

args = parser.parse_args()


if __name__ == '__main__':
    root_dir = '../'
    if args.dataset == 1:
        test = pd.read_csv(os.path.join(root_dir, 'split_sample/test.csv'))

    elif args.dataset == 2:
        test = pd.read_csv(os.path.join(root_dir, 'split/test.csv'))
    
    test = prepare_data(test,outlier=args.outlier)
    
    modeling_pipeline = load_model("Ridge_Model.pkl")
    
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
