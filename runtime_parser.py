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
        print "-d degree used in the svm classifier"
        print "-r the coef0 to be used in the svm classifier"
        print "-p if you want the probabilities in SVM to be calculated"
        print "NOTE: The default runtime parameters are (-cv kf 5 -m 1 -c 1 -gm 1 -d 3 -r 1)"
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
cv = (argvs[argvs.index("-cv") + 1],argvs[argvs.index("-cv") + 2]) if "-cv" in argvs else ("kf",5)
classifier = int(argvs[argvs.index("-m") + 1]) if "-m" in argvs else 1
C = int(argvs[argvs.index("-c") + 1]) if "-c" in argvs else 1
gamma = float(argvs[argvs.index("-gm") + 1]) if "-gm" in argvs else 1
r = float(argvs[argvs.index("-r") + 1]) if "-r" in argvs else 1
degree = int(argvs[argvs.index("-d") + 1]) if "-d" in argvs else 3
probabilities = True if "-p" in argvs else False