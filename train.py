import argparse
import joblib
from data_helper import*
import os
from utilities import*
from regression_utils import*
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=2,
                    help =  '1 for split_sample'
                            '2 for full dataset')

parser.add_argument('--outlier', type=bool, default=False)

parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                    help='MinMaxScaler for min/max scaling and StandardScaler for standardizing')

parser.add_argument('--weak_features_drop',type =bool,default=False)

parser.add_argument('--search_best_params',type =bool,default=False)

parser.add_argument('--saveModel',type =bool,default=True)

args = parser.parse_args()


if __name__ == '__main__':
    root_dir = '../'
    if args.dataset == 1:
        df1 = pd.read_csv(os.path.join(root_dir, 'split_sample/train.csv'))
        df2 = pd.read_csv(os.path.join(root_dir, 'split_sample/val.csv'))
        df3 = pd.read_csv(os.path.join(root_dir, 'split_sample/test.csv'))

        # Concatenate them into one dataframe
        df = pd.concat([df1, df2, df3], ignore_index=True)

        # Split into train and test
        train, test = train_test_split(
            df, 
            test_size=0.2, 
            shuffle=True, 
            random_state=17
        )

    elif args.dataset == 2:
        df1 = pd.read_csv(os.path.join(root_dir, 'split/train.csv'))
        df2 = pd.read_csv(os.path.join(root_dir, 'split/val.csv'))
        df3 = pd.read_csv(os.path.join(root_dir, 'split/test.csv'))

        # Concatenate them into one dataframe
        df = pd.concat([df1, df2, df3], ignore_index=True)

        # Split into train and test
        train, test = train_test_split(
            df, 
            test_size=0.2, 
            shuffle=True, 
            random_state=17
        )

    train = prepare_data(train,outlier=args.outlier,weak_features_drop = args.weak_features_drop)
    test = prepare_data(test,outlier=args.outlier,weak_features_drop=args.weak_features_drop)
    
    print(args)
    
    args_dict = vars(args)
    model_args = {
        "preprocessing":args_dict['preprocessing']
    }
    alphas = [0.01,0.1,1,10,100]
    
    best_alpha = 100
    best_degree = 5
    
    if args.search_best_params:
        model,numeric_features,categorical_features = find_best_degree_alpha(train,alphas,model_args) #Best alpha =100 and degree = 5
    
    else:
        model,numeric_features,categorical_features = train_model(train,best_alpha=best_alpha,best_degree=best_degree,args=model_args)
    
    evaluate(model,test,numeric_features+categorical_features,"Test")
    
    if args.saveModel:
        
        filtered_args = {
            "weak_features_drop": args_dict["weak_features_drop"],
            "outlier": args_dict["outlier"]
        }
        
        model_data = {
        'model': model,
        'Polynomial_Degree':best_degree,
        'Alpha': best_alpha,
        'Scaler':args.preprocessing,
        'Outlier_Removal': args.outlier,
        'Numerical features':numeric_features,
        'Categorical features': categorical_features,
        'args': filtered_args
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

# Namespace(dataset=1, outlier=False, preprocessing='StandardScaler', degree=4, weak_features_drop=False, saveModel=False)
# Train has R2 score: 0.6452331503123522 and RMSE: 0.477202876785712
# Test has R2 score: 0.585099744498067 and RMSE: 0.4755554948858555

# Namespace(dataset=1, outlier=False, preprocessing='StandardScaler', degree=4, weak_features_drop=True, saveModel=False)
# Train has R2 score: 0.6301552750835019 and RMSE: 0.48723811209106965
# Test has R2 score: 0.587927334831535 and RMSE: 0.4739322433076791

# Namespace(dataset=1, outlier=False, preprocessing='StandardScaler', degree=6, weak_features_drop=True, saveModel=False)
# Train has R2 score: 0.6325516951638476 and RMSE: 0.48565700968707776
# Test has R2 score: 0.5917104158917128 and RMSE: 0.4717517321007779

# Namespace(dataset=1, outlier=False, preprocessing='StandardScaler', degree=6, weak_features_drop=False, saveModel=False)
# Train has R2 score: 0.6525055723266242 and RMSE: 0.47228642173086144
# Test has R2 score: 0.5906101548922225 and RMSE: 0.4723869440166838

#-------------------------------------------------------

# Namespace(dataset=2, outlier=False, preprocessing='StandardScaler', degree=4, weak_features_drop=False, saveModel=False)
# Train has R2 score: 0.6653862555838408 and RMSE: 0.45972568485981974
# Test has R2 score: 0.6374749413097359 and RMSE: 0.4792135885057625

# Namespace(dataset=2, outlier=False, preprocessing='StandardScaler', degree=4, weak_features_drop=True, saveModel=False)
# Train has R2 score: 0.6532404509561935 and RMSE: 0.4679948729708679
# Test has R2 score: 0.6558526669475873 and RMSE: 0.4669090724810242

# Namespace(dataset=2, outlier=False, preprocessing='StandardScaler', degree=6, weak_features_drop=True, saveModel=False)
# Train has R2 score: 0.6551077907790722 and RMSE: 0.46673306898540146
# Test has R2 score: 0.6575285971629057 and RMSE: 0.4657708070669033