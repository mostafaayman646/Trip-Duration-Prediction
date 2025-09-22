import pandas as pd
import numpy as np

def encoding(df_train,df_val,df_test):
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = ['passenger_count', 'months', 'day_of_week', 'hours']

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.fit(df_train[cat_cols])

    # Transform and create DataFrames for encoded features
    train_encoded = pd.DataFrame(enc.transform(df_train[cat_cols]), columns=enc.get_feature_names_out(cat_cols), index=df_train.index)
    val_encoded = pd.DataFrame(enc.transform(df_val[cat_cols]), columns=enc.get_feature_names_out(cat_cols), index=df_val.index)
    test_encoded = pd.DataFrame(enc.transform(df_test[cat_cols]), columns=enc.get_feature_names_out(cat_cols), index=df_test.index)

    # Drop original categorical columns
    df_train = df_train.drop(columns=cat_cols)
    df_val = df_val.drop(columns=cat_cols)
    df_test = df_test.drop(columns=cat_cols)

    # Concatenate the encoded columns
    df_train = pd.concat([df_train, train_encoded], axis=1)
    df_val = pd.concat([df_val, val_encoded], axis=1)
    df_test = pd.concat([df_test, test_encoded], axis=1)
    
    #Make target at last again
    log_trip_duration = df_train.pop('log_trip_duration')
    df_train['log_trip_duration'] = log_trip_duration
    log_trip_duration = df_val.pop('log_trip_duration')
    df_val['log_trip_duration'] = log_trip_duration
    log_trip_duration = df_test.pop('log_trip_duration')
    df_test['log_trip_duration'] = log_trip_duration

    return df_train, df_val, df_test

def outlier_removal(df,features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

def deg2rad(deg):
    return np.deg2rad(deg)

def getDistanceFromLatLonInM(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Earth radius in meters
    
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1)
    
    a = np.sin(dLat / 2) ** 2 + np.cos(deg2rad(lat1)) * np.cos(deg2rad(lat2)) * np.sin(dLon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def to_numpy(df,idx = -1):
    data = df.to_numpy()
    
    if idx == -1:
        X = data[:,:-1]
    
    else:
        X = data[:,idx].reshape(-1,1)
    
    t = data[:,-1].reshape(-1,1)
    
    return data,X,t

def load_data(data_path,outlier = False):
    df = pd.read_csv(data_path)
    
    #Datetime features
    df['pickup_datetime'] = df['pickup_datetime'].astype('datetime64[ns]')
    months = df['pickup_datetime'].dt.month.astype(int)
    day_of_week = df['pickup_datetime'].dt.day_of_week.astype(int)
    hours = df['pickup_datetime'].dt.hour.astype(int)
    df.drop(columns='pickup_datetime',inplace=True)
    df = df.assign(
        months = months,
        day_of_week = day_of_week,
        hours = hours
    )
    
    #Removing id columns
    df.drop(columns=['id','vendor_id'], inplace=True)
    
    #Changing store flag to int
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).astype('int64')
    
    #Outliers
    if outlier:
        features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']
        df = outlier_removal(df,features)
    
    #Distance feature
    lat1 = df['pickup_latitude'].to_numpy()
    lon1 = df['pickup_longitude'].to_numpy()
    lat2 = df['dropoff_latitude'].to_numpy()
    lon2 = df['dropoff_longitude'].to_numpy()
    distance = getDistanceFromLatLonInM(lat1, lon1, lat2, lon2)
    df = df.assign(
        Distance = distance
    )
    
    #Log transformation
    df['log_trip_duration'] = np.log1p(df.trip_duration)
    df.drop(columns='trip_duration',inplace=True)
    
    #Remove duplicates
    df = df.drop_duplicates(keep="last")
    
    return df