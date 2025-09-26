import argparse
from data_helper import*
from utilities import*
from regression_utils import*


parser = argparse.ArgumentParser()

parser.add_argument('--outlier', type=bool, default=False)

parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                    help='MinMaxScaler for min/max scaling and StandardScaler for standardizing')

parser.add_argument('--weak_features_drop',type =bool,default=False)

parser.add_argument('--search_best_params',type =bool,default=False)

parser.add_argument('--Alpha',type = int,default=100)
parser.add_argument('--Degree',type = int,default=5)

parser.add_argument('--saveModel',type =bool,default=True)

parser.add_argument('--test_size',type =int,default=0.2)

parser.add_argument('--Shuffle',type =bool,default=True)

args = parser.parse_args()


if __name__ == '__main__':
    #Loading data
    train, test = load_data_from_zip('Data/split.zip',test_size = args.test_size,Shuffle=args.Shuffle)

    #Preparing Data
    train = prepare_data(train,outlier=args.outlier,weak_features_drop = args.weak_features_drop)
    test = prepare_data(test,outlier=args.outlier,weak_features_drop=args.weak_features_drop)
    
    print(args)
    
    alphas = [0.01,0.1,1,10,100] #For investigation only
    
    
    if args.search_best_params:
        model,numeric_features,categorical_features = find_best_degree_alpha(train,alphas,args.preprocessing) #Best alpha =100 and degree = 5
    
    else:
        #Training model
        model,numeric_features,categorical_features = train_model(train,best_alpha=args.Alpha,best_degree=args.Degree,processor= args.preprocessing)
    
    evaluate(model,test,numeric_features+categorical_features,"Test")
    
    if args.saveModel:
        save_configs_and_model(model, numeric_features, categorical_features, args)