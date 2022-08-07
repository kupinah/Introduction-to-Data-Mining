from calendar import month_name
from datetime import datetime
import matplotlib.pyplot as plt
from random import random
import numpy as np
import pandas as pd
import pickle
from pyparsing import col
from sklearn import metrics
import sys
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from lpputils import tsadd, tsdiff, parsedate
from linear import *
np.set_printoptions(threshold=sys.maxsize)

def float_to_datetime(f):
    return datetime.fromtimestamp(f).strftime('%Y-%m-%d %H:%M:%S.%f')

def datetime_to_float(d):
    return d.timestamp()

def data_intersection(X, Y, train):
    registrations_train = pd.get_dummies(X["Registration"])
    drivers_train = pd.get_dummies(X["Driver ID"])
    
    registrations_test = pd.get_dummies(Y["Registration"])
    drivers_test = pd.get_dummies(Y["Driver ID"])

    reg_intersection = registrations_test.columns.intersection(registrations_train.columns)
    driver_intersection = drivers_test.columns.intersection(drivers_train.columns)

    if(train):
        return registrations_train[reg_intersection], drivers_train[driver_intersection]
    else:
        return registrations_test[reg_intersection], drivers_test[driver_intersection]

def divide_by_hours_and_days(data, train):
    matrix = np.zeros((len(data), 91)) #verzija dan+ura+minute
    
    beginning_of_route = np.zeros(len(data))
    end_of_route = np.zeros(len(data))

    for i in range(len(data)):
        beginning = parsedate(data["Departure time"][i])
        beginning_f = datetime_to_float(beginning)
        beginning_of_route[i] = beginning_f
        day = pd.Timestamp(beginning).day_of_week
        minute = beginning.minute
        matrix[i, beginning.hour] = 1
        matrix[i, 24 + day] = 1 #-- dan ura
        matrix[i, 31 + minute] = 1 #-- dan ura minuta

        if(train):
            end = parsedate(data["Arrival time"][i])
            end = datetime_to_float(end) - beginning_f
            end_of_route[i] = end

    return matrix, end_of_route

def create_model_by_hours_and_days():
    train_data = pd.read_csv("./train_pred.csv", sep='\t')
    test_data = pd.read_csv("./test_pred.csv", sep='\t')
    X, y = divide_by_hours_and_days(train_data, True)

    reg, driv = data_intersection(train_data, test_data, True)

    data = np.concatenate((X, reg, driv), axis=1)

    X_train = data[:8476,:]
    X_test = data[8476:,:]

    y_train = y[:8476]
    y_test = y[8476:]

    lr = LinearLearner(lambda_=1.6)
    model = lr(X_train, y_train)

    y_pred = []
    for item in X_test:
        y_pred.append(model(item))
    
    mae = round(metrics.mean_absolute_error(y_test, y_pred), 3)
    print(mae)
    
    if(mae < 135):
        pickle.dump(model, open('model', 'wb'))

def load_model():
    train_data = pd.read_csv("./train_pred.csv", sep='\t')
    test_data = pd.read_csv("./test_pred.csv", sep='\t')

    X, _ = divide_by_hours_and_days(test_data, False)
    reg, driv = data_intersection(train_data, test_data, False)
    data = np.concatenate((X, reg, driv), axis=1)

    model = pickle.load(open('model', 'rb'))

    prediction = []
    for i in range(len(test_data)):
        prediction.append(model(data[i,:]))

    f = open("output.txt", "w")
    for i in range(len(prediction)):
        output = tsadd(test_data["Departure time"][i], prediction[i])
        f.write(output + "\n")
    f.close()

create_model_by_hours_and_days()
load_model()