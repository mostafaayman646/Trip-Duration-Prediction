def ridge():
    from sklearn.linear_model import  Ridge
    
    return Ridge(fit_intercept=True)

def lasso(alpha):
    from sklearn.linear_model import Lasso
    
    return  Lasso(fit_intercept = True, alpha = alpha, max_iter = 10000)

def evaluate(x_train, t_train,x_val, t_val, model, name):
    from sklearn.metrics import r2_score
    import numpy as np
    
    print(f"results of :",name)
    
    model.fit(x_train,t_train)
    
    avg_abs_weght = abs(model.coef_).mean()
    print(f'\tintercept: {model.intercept_} - abs avg weight: {avg_abs_weght}')
    
    pred_t_train = model.predict(x_train)
    
    r2_train = r2_score(t_train,pred_t_train)
    print(f"Train err: {r2_train}")
    
    pred_t_val = model.predict(x_val)
    
    r2_val = r2_score(t_val,pred_t_val)
    print(f"Val err: {r2_val}")
    
    return r2_train,r2_val

def InvestigateRidgeParams(X,t,option=1):
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    
    if option == 1:
        pipeline = make_pipeline(MinMaxScaler(),
                             ridge())
    
    elif option == 2:
        pipeline = make_pipeline(StandardScaler(),
                             ridge())

    kf = KFold(n_splits=4, random_state=35, shuffle=True)
    scores = cross_val_score(pipeline, X, t, cv = kf,
                             scoring = 'r2')
    print(scores)
    print(scores.mean(), scores.std())

def lassoSelection(x_train,x_val,t_train,t_val,alphas):
    from sklearn.feature_selection import SelectFromModel
    import numpy as np
    from sklearn.metrics import r2_score
    
    alphas_dict = {}
    
    selected_features = {}
    
    for alpha in alphas:
        model = lasso(alpha)
        
        sfm = SelectFromModel(model )
        sfm.fit(x_train,t_train)
        
        idx = sfm.get_support()
        
        
        pred_t = sfm.estimator_.predict(x_val)
        lass_val_err = r2_score(t_val, pred_t)
        
        selected_features[alpha] = idx
        alphas_dict[alpha] = lass_val_err
        
        
        print(f'alpha={alpha}, selects {len(np.flatnonzero(idx))} features and has {lass_val_err} val error')
    
    print('+++++++++++++++++++++++++++++++++++++++++++++\n')
    
    return selected_features[0.01]