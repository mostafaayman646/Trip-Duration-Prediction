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
    drop_features = ['vendor_id','pickup_longitude','pickup_latitude',
                     'dropoff_longitude','dropoff_latitude','direction']
    
    return df.drop(columns = drop_features)

def prepare_data(df,outlier,weak_features_drop):
    
    #Datetime features
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"]) 
    
    # Hour bins (Time of day)
    hour_bins   = [0, 6, 12, 16, 21, 24]   # cover full 24 hours
    hour_labels = ['Night', 'Morning', 'Afternoon', 'Evening', 'Late Night']
    df["hour_period_bin"] = pd.cut(
        df["pickup_datetime"].dt.hour,
        bins=hour_bins,
        labels=hour_labels,
        right=False
    )

    # Day of week: weekday vs jobdays
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["dayofweek"] = df["dayofweek"].map(
        lambda x: "Weekend" if x >= 5 else "Jobday"
    )

    # Day of month and day of year
    df["day"] = df["pickup_datetime"].dt.day

    doy_bins   = [0, 90, 180, 270, 365]  
    doy_labels = ["Q1", "Q2", "Q3", "Q4"]  

    df["dayofyear_bin"] = pd.cut(
        df["pickup_datetime"].dt.day_of_year,
        bins=doy_bins,
        labels=doy_labels
    )

    # Month and # Seasons
    df["month"] = df["pickup_datetime"].dt.month

    season_bins   = [0, 3, 6, 9, 11]   # Winter(Dec–Feb), Spring, Summer, Autumn
    season_labels = ['Winter', 'Spring', 'Summer', 'Autumn']
    df["season"] = pd.cut(
        df["month"],
        bins=season_bins,
        labels=season_labels,
        right=False
    )
    
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