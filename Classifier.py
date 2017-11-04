from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas as pd
import numpy as np


def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    n_train = int(np.floor(x.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]

    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]

    return x_train, y_train, x_val, y_val


dataset = pd.read_csv("Wholesale customers data.csv")
X = dataset.iloc[:,2:].values
Y = dataset.iloc[:,0].values
particions = [0.5, 0.7, 0.8]

# LOGISTIC
for part in particions:
    x_t, y_t, x_v, y_v = split_data(X, Y, part)

    # Creation of logistic regresor
    logreg = LogisticRegression()

    # Training
    logreg.fit(x_t, y_t)
    y_pred = logreg.predict(x_v)
    percent_correct_log = np.mean(y_v == y_pred).astype('float32')
    print "Correct classification logistic ", part, "%: ", percent_correct_log, "\n"

# SVM
for part in particions:
    x_t, y_t, x_v, y_v = split_data(X, Y, part)
    svc = svm.SVC(C=100,kernel='linear',gamma=0.001,probability=True)
    svc.fit(x_t, y_t)
    y_pred = svc.predict(x_v)
    
    percent_correct_svm = np.mean(y_v == y_pred).astype('float32')
    print "Correct classification svm ", part, "%: ", percent_correct_svm, "\n"