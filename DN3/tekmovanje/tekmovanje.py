from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from lpputils import tsadd, tsdiff, parsedate
from linear import *
from sklearn.model_selection import train_test_split

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

def parse_prazniki():
    data = pd.read_csv('./prazniki.csv')
    prazniki = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        prazniki[i,0] = datetime.strptime(data["DATUM"][i], '%d.%m.%Y').day
        prazniki[i,1] = datetime.strptime(data["DATUM"][i], '%d.%m.%Y').month
    
    return prazniki

def divide_by_hours_and_days(data, train, prazniki):
    # parse of hours, days, minutes and months
    # method is separated in 2 parts - train and test
    # because train uses a block of data and test uses one row per time
    if train == False:
        matrix = np.zeros((len(data), 105))
        beginning_of_route = np.zeros(len(data))
        for i in range(len(data)):
            beginning = parsedate(data["Departure time"][i])
            beginning_f = datetime_to_float(beginning)
            beginning_of_route[i] = beginning_f
            
            day = pd.Timestamp(beginning).dayofweek
            minute = beginning.minute
            month = beginning.month

            matrix[i, beginning.hour] = 1
            matrix[i, 24 + day] = 1 
            matrix[i, 31 + minute] = 1
            matrix[i, 91 + month] = 1

            check_praznik = np.array([float(beginning.day), float(month)])

            if np.any(np.all(check_praznik == prazniki, axis=1)):
                matrix[i,-1] = 1

    else:
        matrix = np.zeros(105)
        beginning = parsedate(data)
        day = pd.Timestamp(beginning).dayofweek
        minute = beginning.minute
        month = beginning.month

        matrix[beginning.hour] = 1
        matrix[24 + day] = 1 
        matrix[31 + minute] = 1
        matrix[91 + beginning.month] = 1
        check_praznik = np.array([float(beginning.day), float(month)])

        if np.any(np.all(check_praznik == prazniki, axis=1)):
            matrix[-1] = 1

    return matrix

def load_model():
    train_data = pd.read_csv("./train.csv", sep='\t')
    test_data = pd.read_csv("./test.csv", sep='\t')
    prazniki = parse_prazniki()
    X = divide_by_hours_and_days(test_data, False, prazniki)
    reg, driv = data_intersection(train_data, test_data, False)

    data = np.concatenate((X, reg, driv), axis=1)

    prediction = []
    models = {}
    for i in range(len(test_data)):
        model_key = test_data["Route Direction"][i]

        # check if model is already loaded - for speed optimization
        if model_key not in models:
            models[model_key] = pickle.load(open('models/model_' + model_key, 'rb'))

        model = models[model_key]        
        prediction.append(model(data[i,:]))

    f = open("output.txt", "w")

    for j in range(len(prediction)):
        output = tsadd(test_data["Departure time"][j], prediction[j])
        f.write(output + "\n")
    
    f.close()

def prepare_models():
    data = pd.read_csv('train.csv', sep = '\t')
    t_data = pd.read_csv('test.csv', sep='\t')

    # create 2D array (day, month) that represents holidays
    prazniki = parse_prazniki()

    # intersect test and train data so we make sure that only data from both DS is present
    reg, driv = data_intersection(data, t_data, True)

    reg = pd.DataFrame(reg)
    driv = pd.DataFrame(driv)
    data = pd.concat([data, reg, driv], axis=1)
    
    models = {}
    unique_routes = pd.unique(data["Route Direction"])

    for un_routes in range(len(unique_routes)):
        # select only data (rows) that refer to current Route
        routes = (data[data["Route Direction"] == unique_routes[un_routes]]).to_numpy()
        matrix = np.zeros((routes.shape[0], 698))
        y = np.zeros(routes.shape[0])
        for r in range(routes.shape[0]):
            # x1 - Departure time, x2 - all externally added data (1-hot-encode of drivers, registrations, holidays), x3 - Arrival time
            x1 = routes[r, 6]
            x2 = routes[r, 9:]
            x3 = datetime_to_float(parsedate(routes[r, 8]))
            x3 -= datetime_to_float(parsedate(x1))

            X = divide_by_hours_and_days(x1, True, prazniki)
            X = np.concatenate((X, x2))
            matrix[r] = X
            y[r] = x3
     
        matrix = np.concatenate((matrix, y[:,None]), axis=1)
        models[unique_routes[un_routes]] = matrix

    for i in models:
        X_train, X_test, y_train, y_test = train_test_split(models[i][:,:-1], models[i][:,-1], test_size=0.3)

        lr = LinearLearner(lambda_=16.)
        model = lr(X_train, y_train)
        y_pred = []
        for item in X_test:
            y_pred.append(model(item))

        pickle.dump(model, open('models/model_' + i, 'wb'))
        # mae = round(metrics.mean_absolute_error(y_test, y_pred), 3)
        # print(mae)

    return models

prepare_models()
load_model()