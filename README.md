# [NYC-Taxi-Trip-Duration](https://www.kaggle.com/code/sherif31/new-york-city-taxi-trip-duration) 
## Project Overview
This project aims to predict the total ride duration of taxi trips in New York City as part of the NYC Taxi Duration Prediction competition on Kaggle. The model uses data provided by the NYC Taxi and Limousine Commission, including information such as pickup time, geo-coordinates, number of passengers, and other variables.

## Repo structure and File descriptions
```
NYC-Taxi-Trip-Duration
├── README.md
├── requirements.txt
├── (EDA) New York City EDA.ipynb
├── Script.py
├── train.py
├── data_helper.py
├── utilities.py
├── regression_utils
├── Data  
│   └── split.csv
├── Models   
│   └── Ridge_model.pkl
└── config
    ├── model_config
    └── data_config
```
- `README.md`: Contains information about the project, how to set it up, and any other relevant details.
- `requirements.txt`: Lists all the dependencies required for the project to run.
- `(EDA) New York City EDA.ipynb`: Jupyter notebook containing exploratory data analysis on New York City taxi trip duration.
- `Scrip.py`: Python script for loading test data.
- `train.py`: Python script for loading training data and saving the model.
- `utilities.py`: Python script containing helper functions used in other scripts.
- `data_helper.py`: Python script for data preparation (Data Versioning).
- `Ridge_model.pkl`: The baseline model stored as pkl format.
- `regression_utils`: Contains the script used to train the model and select the best alpha and polynomial degree.


## Dependencies
```shell
pip install -r requirements.txt
```
## Usage
Make sure to be in Trip-Duration-Prediction

1. Download the Data folder from the Google Drive link and put it inside the repo folder
2. Run Script.py
## Notes

All the data files are on Google Drive due to GitHub's limitations on pushing larger data files.
Feel free to download it from here [Google Drive](https://drive.google.com/drive/folders/1hzFa7VH7V2SV16pS7FE7-PFchSZboy4n?usp=sharing)

## Data exploration

### Target Variable: Trip Duration
- Distribution resembles a Gaussian distribution with a long right tail (right-skewed)
- Most trips are between 150 seconds and 1000 seconds (about 2.5 to 16.7 minutes)
- Log transformation applied to visualize better and help with modeling large values

### Feature Analysis
1. Discrete Numerical Features:
   - Vendor ID and passenger count analyzed
   - No significant difference in trip duration among vendors
   - Trips with 7-8 passengers tend to have shorter durations, possibly due to the trip purpose

2. Geographical Features:
   - Haversine distance calculated using pickup and dropoff coordinates
   - Most trips range from less than 1 km to 25 km
   - Speed of trips calculated using distance and duration

3. Temporal Analysis:
   - Longer trip durations observed during summer months
   - Weekend trips are generally longer than weekdays
   - Shorter durations during morning and evening rush hours

### Correlation Analysis
- Strong positive correlation between trip duration and distance
- Negative correlation between trip duration and speed

## Modeling

### Data Pipeline
1. Feature splitting into categorical and numerical
2. One-hot encoding for categorical features
3. Standard scaling for numerical features
4. Polynomial Features (degree=5)
5. Log transformation applied

### Results
- Test has R2 score: 0.6238187914490088 and RMSE: 0.4877039854290511