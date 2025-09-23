import argparse
import joblib
from data_helper import*
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from utilities import*

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=2,
                    help =  '1 for split_sample'
                            '2 for full dataset')

parser.add_argument('--outlier', type=bool, default=False)

parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                    help='MinMaxScaler for min/max scaling and StandardScaler for standardizing')

parser.add_argument('--degree', type=int, default=4,
                    help = 'add polynomial degrees of numerical features to your dataframe')

parser.add_argument('--saveModel',type =bool,default=True)

args = parser.parse_args()


def make_pipeline(train,val):
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','distance_haversine'
                        ,'direction','distance_manhattan']
    
    categorical_features = ['dayofweek', 'month', 'hour', 'day', 'Season', 'store_and_fwd_flag', 
                            'passenger_count', 'vendor_id']
    
    train_features = categorical_features + numeric_features
    
    if args.preprocessing == 'MinMaxScaler':
        scaler = MinMaxScaler()
    
    elif args.preprocessing == 'StandardScaler':
        scaler = StandardScaler()
    
    numeric_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=args.degree, include_bias=False)),
        ("scaler", scaler)
    ])
    
    clf = ColumnTransformer([
        ('ohe',OneHotEncoder(handle_unknown='ignore'),categorical_features),
        ('numeric',numeric_pipeline,numeric_features)
        ], 
        remainder='passthrough')
    
    pipeline = Pipeline(steps=[
        ('clf', clf),
        ('regression', Ridge())
    ])
    
    model = pipeline.fit(train[train_features], train.log_trip_duration)
    
    evaluate(model,train,train_features,'Train')
    evaluate(model,val,train_features,'Val')
    
    return model,numeric_features,categorical_features


if __name__ == '__main__':
    root_dir = '../'
    if args.dataset == 1:
        train = pd.read_csv(os.path.join(root_dir, 'split_sample/train.csv'))
        val = pd.read_csv(os.path.join(root_dir, 'split_sample/val.csv'))

    elif args.dataset == 2:
        train = pd.read_csv(os.path.join(root_dir, 'split/train.csv'))
        val = pd.read_csv(os.path.join(root_dir, 'split/val.csv'))

    train = prepare_data(train,outlier=args.outlier)
    val = prepare_data(val,outlier=args.outlier)
    
    model,numeric_features,categorical_features = make_pipeline(train,val)

    if args.saveModel:
        model_data = {
        'model': model,
        'Polynomial_Degree':args.degree,
        'Scaler':args.preprocessing,
        'Outlier_Removal': args.outlier,
        'Numerical features':numeric_features,
        'Categorical features': categorical_features
        }
        
        filename = 'Ridge_Model.pkl'
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")


# Degree = 1:
# Train has R2 score: 0.6245125155357683
# Val has R2 score: 0.6295451765164988
# Degree= 2:
# Train has R2 score: 0.6492037888676272
# Val has R2 score: 0.6480441127359529
# Degree = 3:
# Train has R2 score: 0.6555860491044175
# Val has R2 score: 0.6549516869113852
# Degree = 4:
# Train has R2 score: 0.6616781688406927
# Val has R2 score: 0.6553690145021055
# Degree = 5:
# Train has R2 score: 0.6672483598866302
# Val has R2 score: 0.6521662327698375