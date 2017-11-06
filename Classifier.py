from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
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
def logistic(x_t,y_t,x_v,y_v):
    logreg = LogisticRegression()
    logreg.fit(x_t, y_t)
    y_pred = logreg.predict(x_v)
    
    
    return logreg.score(x_v,y_v)


# SVM
def support(x_t,y_t,x_v,y_v):
    svc = svm.SVC(C=1, kernel='linear', gamma=0.001, probability=True)
    svc.fit(x_t, y_t)
    y_pred = svc.predict(x_v)
    
    return svc.score(x_v,y_v)

# Apply K-Fold Cross Validation
def kfold(x,y,num,method="logistic"):
    kf = KFold(n_splits=num)
    count = 0
    mscore = 0.0
    print "n splits kfold" , num
    
    for train_index, test_index in kf.split(x):
        x_t, x_v = X[train_index], X[test_index]
        y_t, y_v = y[train_index], y[test_index]
        
        if method=="logistic":
            mscore += logistic(x_t,y_t,x_v,y_v)
        if method =='svm':
            mscore += support(x_t,y_t,x_v,y_v)
        count += 1
    mscore = mscore/count
    
    return mscore

# Apply Leave One Out
def oneout(x,y,method="logistic"):
    loo = LeaveOneOut()
    count = 0
    mscore = 0.0
    print "nsplits oneout" , loo.get_n_splits(x)
    
    for train_index, test_index in loo.split(x):
        x_t, x_v = X[train_index], X[test_index]
        y_t, y_v = y[train_index], y[test_index]
        
        if method =="logistic":
            mscore += logistic(x_t,y_t,x_v,y_v)
        else:
            mscore += support(x_t,y_t,x_v,y_v)
        count += 1
    
    mscore = mscore/count
    
    return mscore

#Apply Random Split
def split(x,y,method):
    particions = [0.5, 0.7, 0.8]
    score = 0.0
    print "particions", particions
    
    for part in particions:
        x_t, y_t, x_v, y_v = split_data(x, y, part)
        
        if method == "logistic":
            score= logistic(x_t, y_t, x_v, y_v)
        else:
            score = support(x_t, y_t, x_v, y_v)
        
        print part, score

print kfold(X,Y,10,"logistic")
print oneout(X,Y,"logistic")
split(X,Y,"logistic")

print kfold(X[:,0],Y,10,"svm")
print oneout(X,Y,"svm")
split(X,Y,"svm")


