from re import A
import time
from datetime import datetime
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from lpputils import tsadd, tsdiff, parsedate
from linear import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor as gbr, StackingRegressor
from mlxtend.regressor import StackingCVRegressor

def float_to_datetime(f):
    return datetime.fromtimestamp(f).strftime('%Y-%m-%d %H:%M:%S.%f')

def datetime_to_float(d):
    return d.timestamp()

def data_intersection(X, Y, train):
    registrations_train = pd.get_dummies(X[:,0])
    drivers_train = pd.get_dummies(X[:,1])
    
    registrations_test = pd.get_dummies(Y[:,0])
    drivers_test = pd.get_dummies(Y[:,1])
    
    reg_intersection = registrations_test.columns.intersection(registrations_train.columns)
    driver_intersection = drivers_test.columns.intersection(drivers_train.columns)

    if(train):
        return registrations_train[reg_intersection], drivers_train[driver_intersection]
    else:
        return registrations_test[reg_intersection], drivers_test[driver_intersection]

def parse_prazniki():
    data = pd.read_csv('./prazniki.csv').to_numpy()
    prazniki = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        prazniki[i,0] = datetime.strptime(data[i,0], '%d.%m.%Y').day
        prazniki[i,1] = datetime.strptime(data[i,0], '%d.%m.%Y').month
    
    return prazniki

def divide_by_hours_and_days(data, train, prazniki):
    # parse of hours, days, minutes and months
    # method is separated in 2 parts - train and test
    # because train uses a block of data and test uses one row per time
    if train == False:
        matrix = np.zeros((len(data), 105))
        beginning_of_route = np.zeros(len(data))
        for i in range(len(data)):
            beginning = parsedate(data[i, 6])
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

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       scoring_test=metrics.mean_absolute_error, do_probabilities = False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    best_model = fitted_model.best_estimator_
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    score = scoring_test(y_test_data, pred)
    
    return [best_model, pred, score]

def get_stacking():
	# define the base models
    estimators = [
    ('xgb',  gbr(alpha=0.76)),
    ('rf', RandomForestRegressor(n_jobs=-1, max_depth=10)),
    ('knn', KNeighborsRegressor(leaf_size=10, n_jobs=-1))]

    stack = StackingRegressor(estimators=estimators,
                            final_estimator=Lasso(), cv=3,
                            n_jobs=-1)

    return stack
    
def load_model():
    train_data = pd.read_csv("./train.csv", sep='\t').to_numpy()
    test_data = pd.read_csv("./test.csv", sep='\t').to_numpy()
    prazniki = parse_prazniki()
    X = divide_by_hours_and_days(test_data, False, prazniki)
    reg, driv = data_intersection(train_data, test_data, False)

    data = np.concatenate((X, reg, driv), axis=1)

    prediction = []
    models = {}
    for i in range(len(test_data)):
        model_key = test_data[i, 3]

        # check if model is already loaded - for speed optimization
        if model_key not in models:
            models[model_key] = pickle.load(open('models/model_' + model_key, 'rb'))

        model = models[model_key]        
        prediction.append(model.predict(data[i,:].reshape(1, -1)))

    f = open("output.txt", "w")

    for j in range(len(prediction)):
        output = tsadd(test_data[j,6], prediction[j][0])
        f.write(output + "\n")
    
    f.close()

def prepare_models(model):
    data = pd.read_csv('train.csv', sep = '\t').to_numpy()
    t_data = pd.read_csv('test.csv', sep='\t').to_numpy()

    # create 2D array (day, month) that represents holidays
    prazniki = parse_prazniki()

    # intersect test and train data so we make sure that only data from both DS is present
    reg, driv = data_intersection(data, t_data, True)

    data = np.concatenate((data, reg, driv), axis=1)
	
    models = {}
    unique_routes = np.unique(data[:,3])

    print("kreiram modele")
    for un_routes in range(len(unique_routes)):
        # select only data (rows) that refer to current Route
        routes = (data[data[:, 3] == unique_routes[un_routes]])
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

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = round(metrics.mean_absolute_error(y_test, y_pred), 3)
        print(mae)

        pickle.dump(model, open('models/model_' + i, 'wb'))

    print("testiram modele")

start_time = time.time()
model = get_stacking()
prepare_models(model)
load_model()
print("--- %s seconds ---" % (time.time() - start_time))
