import argparse
from data_helper import*
from utilities import*
from regression_utils import*


parser = argparse.ArgumentParser()

#Models args
parser.add_argument('--data_path',type = str , default = 'Data/split.zip')
parser.add_argument('--model_path',type = str, default='Models/Ridge_Model.pkl')
parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                    help='MinMaxScaler for min/max scaling and StandardScaler for standardizing')
parser.add_argument('--search_best_params',type =bool,default=False)
parser.add_argument('--cv_folds',type =int,default=4)
parser.add_argument('--alphas_to_search',type =int,default=[0.01,0.1,1,10,100,10000])
parser.add_argument('--degrees_to_search',type =int,default=[3,4,5,6,7])
parser.add_argument('--RandomSeed',type =int,default=17)
parser.add_argument('--test_size',type =int,default=0.2)
parser.add_argument('--Alpha',type = int,default=100)
parser.add_argument('--Degree',type = int,default=6)
parser.add_argument('--save_model_config',type =bool,default=True)
parser.add_argument('--outlier', type=bool, default=False)
parser.add_argument('--weak_features_drop',type =bool,default=True)
parser.add_argument('--Shuffle',type =bool,default=True)

args = parser.parse_args()


if __name__ == '__main__':
    #Loading data
    train, test = load_data_from_zip(args.data_path,test_size = args.test_size,Shuffle=args.Shuffle)

    #Preparing Data
    train = prepare_data(train,outlier=args.outlier,weak_features_drop = args.weak_features_drop)
    test = prepare_data(test,outlier=args.outlier,weak_features_drop=args.weak_features_drop)
    
    if args.search_best_params:
        model,numeric_features,categorical_features = find_best_degree_alpha(train,args)
    
    else:
        #Training model
        model,numeric_features,categorical_features = train_model(train,args)
    
    evaluate(model,test,numeric_features+categorical_features,"Test")
    
    if args.save_model_config:
        save_configs_and_model(model, numeric_features, categorical_features, args)
