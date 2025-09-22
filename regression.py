def ridge():
    from sklearn.linear_model import  Ridge
    
    return Ridge()

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

def InvestigateRidgeParams(x,t,alphas,model,option=1):
    import numpy as np
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import  MinMaxScaler,StandardScaler
    
    if option ==1:
        pipeline = Pipeline(steps=[("scaler",MinMaxScaler()),('model',model)])
    
    elif option ==2:
        pipeline = Pipeline(steps=[("scaler",StandardScaler()),('model',model)])
    
    elif option ==0:
        pipeline = Pipeline(steps=[('model',model)])
    
    grid = {'model__alpha': alphas}
    
    kf = KFold(n_splits=4, shuffle=True, random_state=17)
    
    search = GridSearchCV(pipeline, grid, cv=kf, scoring= 'r2')
    
    search.fit(x, t)
    
    r2_score = search.cv_results_['mean_test_score'] 
    
    print('Best Parameters: ',search.best_params_)
    
    print('r2_score',r2_score)
    
    return r2_score,search.best_params_['model__alpha']