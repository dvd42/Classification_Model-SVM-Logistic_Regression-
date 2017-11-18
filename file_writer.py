from __future__ import print_function
import runtime_parser as rp
import os


def create_dir(method, split, classifier):


    path = "Holdout " + str(split) if method == "h" else "k-fold " + str(int(split))
    path += "/SVM" if classifier == 1 else "/Logistic"

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def add_file_header(path):

    print ("Running model with:\n"
           "C: %.2f\n"
           "Gamma: %.2f\n"
           "Degree: %d\n"
           "Coef0: %d\n" % (rp.C,rp.gamma,rp.degree,rp.r), file=open(path + "/Results.txt", "a+"))



def store_score(kernel_score,path):
    for key in kernel_score:
        print ("%s: %.3f" % (key,kernel_score[key]) ,file=open(path + "/Results.txt", "a+"))

    print ("\n",file=open(path + "/Results.txt", "a+"))
