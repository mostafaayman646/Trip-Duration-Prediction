from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from utilities import*
from sklearn.model_selection import RandomizedSearchCV, KFold

def find_best_degree_alpha(train,alphas,processor):
    numeric_features = (
    train.select_dtypes(include=["float64"])
    .columns
    .drop("log_trip_duration")
    .tolist()
    )
    
    categorical_features = (
    train.select_dtypes(exclude=["float64"])
    .columns
    .tolist()
    )
    
    train_features = categorical_features + numeric_features
    
    if processor == 'MinMaxScaler':
        scaler = MinMaxScaler()
    
    elif processor == 'StandardScaler':
        scaler = StandardScaler()
    
    numeric_pipeline = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
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
    
    grid = {'clf__numeric__poly__degree': [4,5],
            'regression__alpha':alphas}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=17)
    
    model = RandomizedSearchCV(pipeline, grid, cv=kf, scoring= 'r2')
    
    model.fit(train[train_features], train.log_trip_duration)
    
    print('Best Parameters: ',model.best_params_)
    
    R2 = model.cv_results_['mean_test_score']
    print('R2 for each alpha:', R2)
    
    return model,numeric_features,categorical_features

def train_model(train,best_alpha,best_degree,processor):
    numeric_features = (
    train.select_dtypes(include=["float64"])
    .columns
    .drop("log_trip_duration")
    .tolist()
    )
    
    categorical_features = (
    train.select_dtypes(exclude=["float64"])
    .columns
    .tolist()
    )
    
    train_features = categorical_features + numeric_features
    
    if processor == 'MinMaxScaler':
        scaler = MinMaxScaler()
    
    elif processor == 'StandardScaler':
        scaler = StandardScaler()
    
    numeric_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=best_degree,include_bias=False)),
        ("scaler", scaler)
    ])
    
    clf = ColumnTransformer([
        ('ohe',OneHotEncoder(handle_unknown='ignore'),categorical_features),
        ('numeric',numeric_pipeline,numeric_features)
        ], 
        remainder='passthrough')
    
    pipeline = Pipeline(steps=[
        ('clf', clf),
        ('regression', Ridge(alpha=best_alpha))
    ])
    
    model = pipeline
    
    model.fit(train[train_features], train.log_trip_duration)
    
    evaluate(model,train,train_features,"Train")
    
    return model,numeric_features,categorical_features