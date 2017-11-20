import sys


def process_runtime_arguments():
    """
    :return: list with runtime parameters
    """

    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print "Usage: main.py, data-set.csv"
        print "-cv cross-validation method to be used (h for holdout and kf for k-fold split) (e.g -cv h 0.8 or -cv kf 10)"
        print "-m method that will be used to classify the data: 1 for svm, 2 logistic regression"
        print "-c the C to be used in the svm classifier"
        print "-gm gamma to be used in the svm classifier"
        print "-ovo whether to use one vs one classification (the default is one vs the rest (ovr))"
        print "-n_clases number of classes in y"
        print "-d degree used in the svm classifier"
        print "-r the coef0 to be used in the svm classifier"
        print "-v will show results on the standard output (if this is not entered the results and plots will be stored in the wd)"
        print "NOTE: The default runtime parameters are (-cv kf 5 -m 1 -c 1 -gm 1 -d 3 -r 1 -ovr -n_clases 2)"
        print "NOTE if you want to use the Leave One Out cross validation method just use -cv kf n"
        print "NOTE The data-set MUST BE the 2nd argument"
        sys.exit(1)

    argvs = []
    for i in range(len(sys.argv)):
        argvs.append(sys.argv[i])

    return argvs

argvs = process_runtime_arguments()

# Get the the values of the runtime parameters
data = argvs[1]
verbose = True if "-v" in argvs else False
ovx =  "ovo" if "-ovo" in argvs else "ovr"
cv = (argvs[argvs.index("-cv") + 1],argvs[argvs.index("-cv") + 2]) if "-cv" in argvs else ("kf",5)
classifier = int(argvs[argvs.index("-m") + 1]) if "-m" in argvs else 1
C = float(argvs[argvs.index("-c") + 1]) if "-c" in argvs else 1
gamma = float(argvs[argvs.index("-gm") + 1]) if "-gm" in argvs else 1
r = float(argvs[argvs.index("-r") + 1]) if "-r" in argvs else 1
degree = int(argvs[argvs.index("-d") + 1]) if "-d" in argvs else 3
n_clases = int(argvs[argvs.index("-n_clases") + 1]) if "-n_clases" in argvs else 2