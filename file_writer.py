from __future__ import print_function
import runtime_parser as rp
import os

# Create directories to store results
def create_dir(method, split, classifier):

    path = "Holdout " + str(split) if method == "h" else "k-fold " + str(int(split))
    path += "/SVM" if classifier == 1 else "/Logistic"

    if not os.path.exists(path):
        os.makedirs(path)

    if not rp.verbose:
        if not os.path.exists(path + "/ROC"):
            os.makedirs(path + "/ROC")

        if not os.path.exists(path + "/Precision-Recall"):
            os.makedirs(path + "/Precision-Recall")

        if not os.path.exists(path + "/Kernels"):
            os.makedirs(path + "/Kernels")

    return path

def add_file_header(path):

    if rp.classifier == 1:
        string = ("Running SVM model with:\n"
            "C: %.2f\n"
            "Gamma: %.2f\n"
            "Degree: %d\n"
            "Coef0: %d\n"
            "Strategy: %s" % (rp.C,rp.gamma,rp.degree,rp.r,rp.ovx))
    else:
        string = "Running Logistic model with"


    if not rp.verbose:
        print (string, file=open(path + "/Results.txt", "a+"))

    print (string)


# Store accuracy score in files
def store_score(kernel_score,path):

    for key in kernel_score:
        print ("%s: %.2f" % (key,kernel_score[key]) ,file=open(path + "/Results.txt", "a+"))

    print ("\n",file=open(path + "/Results.txt", "a+"))


# Store F1-score in files or print them if verbose is False
def store_fscore(key,path,micro=None,macro=None,binary = None):

    if rp.verbose:
        if micro != None and macro != None:
            print ("%s: F1_score macro: %.2f" % (key,macro))
            print ("%s: F1_score micro: %.2f"%(key,micro))
        else:
            print ("%s: F1_score: %.2f" % (key,binary))

    else:
        if micro != None and macro != None:
            print("%s: F1_score macro: %.2f" % (key, macro),file=open(path + "/Results.txt", "a+"))
            print("%s: F1_score micro: %.2f" % (key, micro),file=open(path + "/Results.txt", "a+"))
        else:
            print("%s: F1_score weighted: %.2f" % (key, binary),file=open(path + "/Results.txt", "a+"))