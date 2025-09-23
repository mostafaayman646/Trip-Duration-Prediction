import argparse
import numpy as np
from data_helper import*
from regression import *
from utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=2,
                    help =  '1 for split_sample'
                            '2 for full dataset')

parser.add_argument('--Remove_outlier', type=bool, default=False)

parser.add_argument('--preprocessing', type=int, default=2,
                    help='0 for no processing, 1 for min/max scaling and 2 for standardizing')

parser.add_argument('--single_features', type=str, default='5')

parser.add_argument('--degree', type=int, default=6,
                    help = 'add polynomial degrees of numerical features to your dataframe')

parser.add_argument('--monomial_features', type=bool, default=False)

parser.add_argument('--choice',type = int,default=1,
                    help = '1 for Simple Ridge model with alpha = 1 on all features'
                           '2 for cross validation on ridge with alpha = 1'
                           '3 for lasso feature selection')

args = parser.parse_args()

if args.dataset == 1:
    df_train = prepare_data("../split_sample/train.csv",outlier=args.Remove_outlier)
    df_val = prepare_data("../split_sample/val.csv")
    df_test = prepare_data("../split_sample/test.csv")
    
    df_train = monomials_poly_features(df_train,args.degree,args.monomial_features)
    df_val = monomials_poly_features(df_val,args.degree,args.monomial_features)
    df_test = monomials_poly_features(df_test,args.degree,args.monomial_features)
    
    df_train,df_val,df_test = encoding(df_train,df_val,df_test)
    
    data_train,X_train,t_train = to_numpy(df_train)
    data_val,X_val,t_val       = to_numpy(df_val)
    data_test,X_test,t_test    = to_numpy(df_test)

elif args.dataset == 2:
    df_train = prepare_data("../split/train.csv",outlier=args.Remove_outlier)
    df_val = prepare_data("../split/val.csv")
    df_test = prepare_data("../split/test.csv")
    
    df_train = monomials_poly_features(df_train,args.degree,args.monomial_features)
    df_val = monomials_poly_features(df_val,args.degree,args.monomial_features)
    df_test = monomials_poly_features(df_test,args.degree,args.monomial_features)
    
    df_train,df_val,df_test = encoding(df_train,df_val,df_test)
    
    data_train,X_train,t_train = to_numpy(df_train)
    data_val,X_val,t_val       = to_numpy(df_val)
    data_test,X_test,t_test    = to_numpy(df_test)

#START OF INVESTIGATION
if args.choice ==1:
    X_train,X_test = transform_train_val(X_train, X_test,args.preprocessing)
    
    model = ridge()
    r2_train,r2_val = evaluate(X_train,t_train,X_test,t_test,model,'Ridge with alpha = 1')

if args.choice == 2:
    X = np.vstack((X_train, X_val))
    t = np.vstack((t_train, t_val))
    
    InvestigateRidgeParams(X,t,option=args.preprocessing)

if args.choice == 3:
    X_train,X_val = transform_train_val(X_train, X_val,args.preprocessing)
    
    alphas = [0.01]
    model = ridge()
    
    selected_features = lassoSelection(X_train, X_val, t_train, t_val,alphas)
    
    X_train = X_train[:,selected_features]
    X_val = X_val[:,selected_features]
    
    r2_train,r2_val = evaluate(X_train,t_train,X_val,t_val,model,'Ridge with alpha = 1')