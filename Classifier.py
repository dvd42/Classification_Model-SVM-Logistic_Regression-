from sklearn.linear_model import LogisticRegression
from sklearn import svm
import collections as c

import  runtime_parser as rp
import file_writer as fw
import Validator as v


# Train logistic or svm classifier
def train(x_train,y_train,kernel):

    if rp.classifier == 1:
        classifier = svm.SVC(C=rp.C, kernel=kernel, gamma=rp.gamma,degree=rp.degree,probability=True,decision_function_shape=rp.ovx)
        classifier.fit(x_train, y_train)
        
    else:
        classifier = LogisticRegression(multi_class=rp.ovx)
        classifier.fit(x_train, y_train)
    
    return classifier

#Process data and get results using holdout
def h_validate(x_train,x_test,y_train,y_test,path):

    kernels = ["rbf", "poly", "sigmoid", "linear"] if rp.classifier == 1 else ["logistic"]
    kernel_score = {key: [] for key in kernels}
    models = []
    for key in kernel_score:
        models.append(train(x_train, y_train, key))
        kernel_score[key] = models[-1].score(x_test, y_test)

    if rp.verbose:
        print kernel_score
    else:
        fw.store_score(kernel_score,path)



# Process data and get results using k-fold validation
def kf_validate(x, y, split,path):
    
    kernels = ["rbf","poly","sigmoid","linear"] if rp.classifier == 1 else ["logistic"]
        
    kernel_score = {key:[] for key in kernels}
    
    for train_index, test_index in split:
         x_train, x_test = x[train_index], x[test_index]
         y_train, y_test = y[train_index], y[test_index]
         for key in kernel_score:    
            classifier = train(x_train,y_train,key)
            kernel_score[key].append(classifier.score(x_test,y_test))
        
    for key in kernel_score:
        kernel_score[key] = reduce(lambda x, y: x + y,kernel_score[key]) / len(kernel_score[key])


    v.evaluate(x_train,classifier._predict_proba(x_test),y_test,len(c.Counter(y)),path)

    if rp.verbose:
        print kernel_score
    else:
        fw.store_score(kernel_score,path)



