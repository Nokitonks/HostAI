import pandas as pd
import sklearn
import numpy as np
import csv
import datetime
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

"""
 Predict dine time takes in info about the current table in the restaurants history in order
 to make predictions on how long it will be sat
 
"""
def learn_dine_time(data):

    new_pd = pd.DataFrame()
    date = pd.to_datetime(data['Date'],format="%Y-%m-%d")
    new_pd['day'] = date.dt.day_of_week
    new_pd['table'] = data['Table']
    new_pd['sat_time']= pd.to_datetime(data['Seated_Time'],format="%H:%M:%S")
    new_pd['app_time']= pd.to_datetime(data['Appetizer_Time'],format="%H:%M:%S")
    new_pd['main_time']= pd.to_datetime(data['Main_Time'],format="%H:%M:%S")
    new_pd['dessert_time']= pd.to_datetime(data['Dessert_Time'],format="%H:%M:%S")
    new_pd['check_drop_time']= pd.to_datetime(data['Check_Dropped_Time'],format="%H:%M:%S")
    new_pd['left_time']= pd.to_datetime(data['Time_Left'],format="%H:%M:%S")
    new_pd['dine_time'] = abs(new_pd['sat_time'] - new_pd['left_time']).dt.total_seconds()/60
    new_pd['sat_time'] = abs(new_pd['sat_time'] - new_pd['app_time']).dt.total_seconds()/60
    new_pd['app_time'] = abs(new_pd['app_time'] - new_pd['main_time']).dt.total_seconds()/60
    new_pd['main_time'] = abs(new_pd['main_time'] - new_pd['dessert_time']).dt.total_seconds()/60
    new_pd['dessert_time'] = abs(new_pd['dessert_time'] - new_pd['check_drop_time']).dt.total_seconds()/60
    new_pd['check_drop_time'] = abs(new_pd['check_drop_time'] - new_pd['left_time']).dt.total_seconds()/60
    X = new_pd[['day','table']]
    new_pd.drop(['day','table','left_time'],axis=1,inplace=True)
    y = new_pd
    print(new_pd.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

    XGB_model_instance = XGBRegressor()
    XGB_model_instance.fit(X_train, y_train)
    return XGB_model_instance



if __name__ == '__main__':
    csv = "fake_restaurant_data.csv"

    model = learn_dine_time(pd.read_csv(csv))