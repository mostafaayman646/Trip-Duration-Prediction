import pandas as pd
import numpy as np

def outlier_removal(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df.loc[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

def haversine_distance(df):
    R = 6371.0  # Earth radius in km
    
    lat1 = np.radians(df['pickup_latitude'].values)
    lon1 = np.radians(df['pickup_longitude'].values)
    lat2 = np.radians(df['dropoff_latitude'].values)
    lon2 = np.radians(df['dropoff_longitude'].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c  # distance in km
    return np.log1p(dist)

def calculate_direction(df):
    lat1 = np.radians(df['pickup_latitude'].values)
    lon1 = np.radians(df['pickup_longitude'].values)
    lat2 = np.radians(df['dropoff_latitude'].values)
    lon2 = np.radians(df['dropoff_longitude'].values)

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(y, x))
    bearing = (bearing + 360) % 360  # normalize to [0,360)
    return np.log1p(bearing)

def manhattan_distance(df):
    lat1 = np.radians(df['pickup_latitude'].values)
    lat2 = np.radians(df['dropoff_latitude'].values)
    lon1 = np.radians(df['pickup_longitude'].values)
    lon2 = np.radians(df['dropoff_longitude'].values)

    # Approximate conversion: 1 degree latitude ≈ 111 km
    lat_dist = np.abs(df['pickup_latitude'].values - df['dropoff_latitude'].values) * 111
    # Longitude distance depends on latitude
    lon_dist = np.abs(df['pickup_longitude'].values - df['dropoff_longitude'].values) * 111 * np.cos(lat1)

    return np.log1p(lat_dist + lon_dist)

def weak_features(df):
    #Based on EDA these are the weak features
    drop_features = ['pickup_latitude','dropoff_latitude','day','dayofweek']
    
    return df.drop(columns = drop_features)

def prepare_data(df,outlier,weak_features_drop):
    
    #Datetime features
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"]) 
    bins = [0, 2, 5, 8, 11, 12]  # 0, 2, 5, 8, 11, 12 represent the starting and ending months of each season
    labels = ['0', '1', '2', '3', '4'] # Labels for each season ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'] 
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"]  = df["pickup_datetime"].dt.day
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"]  = df["pickup_datetime"].dt.month
    df['Season'] = pd.cut(df["month"] , bins=bins, labels=labels, right=False,ordered=False) 

    #Outliers
    if outlier:
        features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']
        df = outlier_removal(df,features) 
    
    #Distance feature
    df['distance_haversine'] = haversine_distance(df)
    df['direction']          = calculate_direction(df)
    df['distance_manhattan'] = manhattan_distance(df)
    
    #Log transformation
    df['log_trip_duration'] = np.log1p(df.trip_duration)
    
    #Removing id,datedtime columns
    df.drop(columns=['id','pickup_datetime','trip_duration'], inplace=True)
    
    if weak_features_drop:
        df = weak_features(df)
    
    log_trip_duration = df.pop('log_trip_duration')
    df['log_trip_duration'] = log_trip_duration
    
    df.reset_index(drop=True, inplace=True)
    
    return df