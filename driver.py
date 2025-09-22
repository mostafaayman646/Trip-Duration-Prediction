import argparse
import numpy as np
from data_helper import*
from regression import *
from utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=2,
                    help =  '1 for split_sample'
                            '2 for full dataset')

parser.add_argument('--Remove_outlier', type=bool, default=True)

parser.add_argument('--oneHotEncoding', type=bool, default=True)

parser.add_argument('--preprocessing', type=int, default=1,
                    help='0 for no processing, 1 for min/max scaling and 2 for standrizing')

parser.add_argument('--single_features', type=str, default='5')

parser.add_argument('--choice',type = int,default=2,
                    help = 'Simple Ridge model with alpha = 1 on all features' 
                    '1 for investigating different polynomial degrees'
                            '2 investigating single features')

args = parser.parse_args()

if args.dataset == 1:
    df_train = load_data("../split_sample/train.csv",outlier=args.Remove_outlier)
    df_val = load_data("../split_sample/val.csv")
    df_test = load_data("../split_sample/test.csv")
    
    if args.oneHotEncoding:
        df_train,df_val,df_test = encoding(df_train,df_val,df_test)
    
    data_train,X_train,t_train = to_numpy(df_train)
    data_val,X_val,t_val       = to_numpy(df_val)
    data_test,X_test,t_test    = to_numpy(df_test)

elif args.dataset == 2:
    df_train = load_data("../split/train.csv",outlier=args.Remove_outlier)
    df_val = load_data("../split/val.csv")
    df_test = load_data("../split/test.csv")
    
    if args.oneHotEncoding:
        df_train,df_val,df_test = encoding(df_train,df_val,df_test)
    
    data_train,X_train,t_train = to_numpy(df_train)
    data_val,X_val,t_val       = to_numpy(df_val)
    data_test,X_test,t_test    = to_numpy(df_test)


if args.choice ==1:
    X_train,X_val = transform_train_val(X_train, X_val,args.preprocessing)
    
    model = ridge()
    r2_train,r2_val = evaluate(X_train,t_train,X_val,t_val,model,'Ridge with alpha = 1')

if args.choice == 2:
    int_array = np.fromstring(args.single_features, dtype=int, sep=',')
    
    x = []
    y1 = []
    y2 = []
    model = ridge()
    for idx in int_array:
        x.append(idx)
        
        data_train,X_train,t_train = to_numpy(df_train,idx)
        data_val,X_val,t_val       = to_numpy(df_val,idx)
            
        X_train,X_val = transform_train_val(X_train, X_val,args.preprocessing)
        
        r2_train,r2_val = evaluate(X_train,t_train,X_val,t_val,model,'Ridge')
        y1.append(r2_train)
        y2.append(r2_val)
    
    bar_plot(x,y1,y2)

