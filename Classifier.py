from sklearn.linear_model import LogisticRegression
from sklearn import svm


# Train logistic or svm classificator
def train(x_train,y_train,method,C,kernel,gamma,degree,probability):

    if method == "svm":

        classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma,degree=degree,verbose=False,probability=probability)
        classifier.fit(x_train, y_train)
        
    else:
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
    
    return classifier

# Process data and get results 
def validate(x,y,split,method):
    
    kernels = ["rbf","poly","sigmoid","linear"] if method == "svm" else ["logistic"]
        
       
        
    kernel_score = {key:[] for key in kernels}
    
    for train_index, test_index in split:
         x_train, x_test = x[train_index], x[test_index]
         y_train, y_test = y[train_index], y[test_index]
         for key in kernel_score:    
            classifier = train(x_train,y_train,method,1,key,20,3,False)
            kernel_score[key].append(classifier.score(x_test,y_test))
        
    for key in kernel_score:
        kernel_score[key] = reduce(lambda x, y: x + y,kernel_score[key]) / len(kernel_score[key])


    return kernel_score



