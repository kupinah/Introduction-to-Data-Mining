import re
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

def softmax(parameters, X):
    parameters = np.reshape(parameters, (X.shape[1], int(parameters.shape[0]/X.shape[1])))
    predictions = np.exp(np.dot(parameters.T, X.T).astype(float))    
    predictions /= np.sum(predictions, axis=0)

    return predictions.T

def cost(parameters, X, y, lambda_):
    reg = lambda_ * np.sum(parameters**2)
    preds = softmax(parameters, X)
    res = np.sum(np.log(preds[np.arange(len(y)), y])) - reg
    return res
    

def onehot(y):
    onehot = np.zeros((len(np.unique(y)), len(y)))
    for i in range(len(y)):
        onehot[y[i], i] = 1

    return onehot.T 


def grad(parameters, X, y, lambda_):
    predictions = softmax(parameters, X)
    encoded = onehot(y)
    res = np.dot(X.T, (encoded - predictions))
    return res.flatten() - (lambda_ * parameters.flatten())

def bfgs(X, y, lambda_):
    # tukaj inicirajte parametere modela
    x0 = np.zeros((X.shape[1], len(np.unique(y))))

    # preostanek funkcije pustite kot je
    res = minimize(lambda pars, X=X, y=y, lambda_=lambda_: -cost(pars, X, y, lambda_),
                   x0,
                   method='L-BFGS-B',
                   jac=lambda pars, X=X, y=y, lambda_=lambda_: -grad(pars, X, y, lambda_),
                   tol=0.00001)
    return res.x


class SoftMaxLearner:

    def __init__(self, lambda_=0, intercept=True):
        self.intercept = intercept
        self.lambda_ = lambda_

    def __call__(self, X, y):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        pars = bfgs(X, y, self.lambda_)
        return SoftMaxClassifier(pars, self.intercept)


class SoftMaxClassifier:

    def __init__(self, parameters, intercept):
        self.parameters = parameters
        self.intercept = intercept

    def __call__(self, X):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        ypred = softmax(self.parameters, X)
        return ypred


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(SoftMaxLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = c(X)
    return results


def test_cv(learner, X, y, k=5):
    fold_size = int(len(y) / k)
    
    randomize = np.arange(len(y))
    random.shuffle(randomize)
    
    X_random = []
    y_random = []

    for ix in randomize:
        X_random.append(X[ix])
        y_random.append(y[ix])

    predictions = []
  
    for i in range(k):
        trainX = []
        trainy = []

        test_fold_start = i * fold_size
        test_fold_end = test_fold_start + fold_size

        if i == k-1:
            testX = X_random[test_fold_start:]
            trainX = X_random[:test_fold_start]
            trainy = y_random[:test_fold_start]
        else:
            testX = X_random[test_fold_start:test_fold_end]
            trainX = X_random[:test_fold_start] + X_random[test_fold_end:]
            trainy = y_random[:test_fold_start] + y_random[test_fold_end:]

        classifier = learner(np.asarray(trainX), np.asarray(trainy))
       
        for el in testX:
            pred = classifier([np.asarray(el)])
            predictions.append(pred[0])
                    
    final = []

    ind = [np.where(randomize == i) for i in range(len(y))]
    for ix in ind:
        final.append(predictions[ix[0][0]])
    
    return np.asarray(final)
        
    
def CA(real, predictions):
    true_pred = 0

    for i in range(len(real)):
        if(np.argmax(predictions[i]) == real[i]):
            true_pred += 1

    return true_pred/len(real)


def log_loss(real, predictions):
    return -np.mean(np.log(predictions[np.arange(len(real)), real]))


def get_best_lambda(train, pred_class):
    
    best_lambda = -1
    best_loss = 10000
    for i in range(20, 50, 5):
        val = i/100
        pred = test_cv(SoftMaxLearner(val), train, pred_class, 5)    
        ca = CA(pred_class, pred)
        print(ca)
        loss = log_loss(pred_class, pred)


        if(loss < best_loss):
            best_loss = loss
            best_lambda = i

    return best_lambda

def create_final_predictions():
    train = pd.read_csv('train.csv', index_col=0).to_numpy()
    test = pd.read_csv('test.csv', index_col=0).to_numpy()

    pred_class = []
    for el in train[:,-1]:
        pred_class.append(int(re.findall(r'\d+', el)[0])-1)

    train = train[:,:-1].astype('int')

    #lmbd = get_best_lambda(train, pred_class) - TO GET BEST LAMBDA - COMMENTED DUE TO SPEED 
    start = time.time()
    lmbd = 0.3
    learner = SoftMaxLearner(lambda_=lmbd)
    model = learner(train, pred_class)

    f = open("final.txt", "w")
    f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
    
    for i in range(test.shape[0]):
        output = str(i+1)
        preds = model([np.asarray(test[i,:])])[0]
        for j in range(len(preds)):
            output += "," + str(preds[j])
        output += "\n"
        f.write(output)
    
    f.close

    print(time.time() - start)

if __name__ == "__main__":
    create_final_predictions()
    