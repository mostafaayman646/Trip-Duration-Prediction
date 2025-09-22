import argparse
import numpy as np
from data_helper import*


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default=2,
                    help =  '1 for split_sample'
                            '2 for full dataset')

parser.add_argument('--Remove_outlier', type=bool, default=True)

parser.add_argument('--oneHotEncoding', type=bool, default=True)

parser.add_argument('--preprocessing', type=int, default=1,
                    help='0 for no processing, 1 for min/max scaling and 2 for standrizing')

parser.add_argument('--single_features', type=str, default='0,3,6')

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


print(df_train.shape)
print(df_val.shape)
print(df_test.shape)