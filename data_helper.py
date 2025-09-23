import pandas as pd
import numpy as np
from geopy import distance
from geopy.point import Point
import math

def fit_transform(data,choice):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler

    if choice == 0:
        return data,None
    
    if choice ==1:
        processor = MinMaxScaler()
        return processor.fit_transform(data), processor
    
    if choice ==2:
        processor = StandardScaler()
        return processor.fit_transform(data), processor

def transform_train_val(X_train, X_val,choice):
    if choice !=0:
        X_train, X_train_transformer = fit_transform(X_train,choice)
        X_val = X_train_transformer.transform(X_val)
    
    return X_train, X_val

def monomials_poly_features(df, degree = 1, monomial_features = False):
    from sklearn.preprocessing import PolynomialFeatures
    
    numerical_features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','distance_haversine',
                          'direction','distance_manhattan']
    
    X = df[numerical_features].to_numpy()
    
    if monomial_features:
        columns = X.shape[1]
        feature_names = numerical_features.copy()
        
        for i  in range(2,degree+1):
            for j, col in enumerate(numerical_features):
                X = np.column_stack((X, X[:, j]**i))
                feature_names.append(f"{col}^{i}")
    
    else:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        X = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(numerical_features)
    
    # Create dataframe with new features
    df_poly = pd.DataFrame(X, columns=feature_names, index=df.index)
    
    return pd.concat([df.drop(columns=numerical_features), df_poly], axis=1)

def encoding(df_train,df_val,df_test):
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = ['passenger_count', 'month', 'dayofweek', 'hour', 'day', 'Season']

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

def haversine_distance(row):
    pick = Point(row['pickup_latitude'], row['pickup_longitude'])
    drop = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    dist = distance.geodesic(pick, drop)
    return np.log1p(dist.km)

def calculate_direction(row):
    pickup_coordinates =  Point(row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    
    # Calculate the difference in longitudes
    delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
    
    # Calculate the bearing (direction) using trigonometry
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
    x = math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) - \
        math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) * \
        math.cos(math.radians(delta_longitude))
    
    # Calculate the bearing in degrees
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    
    # Adjust the bearing to be in the range [0, 360)
    bearing = (bearing + 360) % 360
    
    return bearing

def manhattan_distance(row):
    lat_distance = abs(row['pickup_latitude'] - row['dropoff_latitude']) * 111  # approx 111 km per degree latitude
    lon_distance = abs(row['pickup_longitude'] - row['dropoff_longitude']) * 111 * math.cos(math.radians(row['pickup_latitude']))  # adjust for latitude
    
    return lat_distance + lon_distance

def to_numpy(df,idx = -1):
    data = df.to_numpy()
    
    if idx == -1:
        X = data[:,:-1]
    
    else:
        X = data[:,idx].reshape(-1,1)
    
    t = data[:,-1].reshape(-1,1)
    
    return data,X,t

def prepare_data(data_path,outlier = False):
    df = pd.read_csv(data_path)
    
    #Datetime features
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"]) 
    bins = [0, 2, 5, 8, 11, 12]  # 0, 2, 5, 8, 11, 12 represent the starting and ending months of each season
    labels = ['0', '1', '2', '3', '4'] # Labels for each season ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'] 
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"]  = df["pickup_datetime"].dt.day
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"]  = df["pickup_datetime"].dt.month
    df['Season'] = pd.cut(df["month"] , bins=bins, labels=labels, right=False,ordered=False) 

    #Changing store flag to int
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).astype('int64')
    
    #Outliers
    if outlier:
        features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']
        df = outlier_removal(df,features)
    
    #Distance feature
    df['distance_haversine'] = df.apply(haversine_distance, axis=1)
    df['direction']          = df.apply(calculate_direction, axis=1)
    df['distance_manhattan'] = df.apply(manhattan_distance, axis=1)
    
    #Log transformation
    df['log_trip_duration'] = np.log1p(df.trip_duration)
    
    #Removing id,datedtime columns
    df.drop(columns=['id','vendor_id','pickup_datetime','trip_duration'], inplace=True)
    
    #Remove duplicates
    df = df.drop_duplicates(keep="last")
    df.reset_index(drop=True, inplace=True)
    
    return df