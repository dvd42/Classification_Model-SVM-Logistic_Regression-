from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from matplotlib import pyplot as plt
from scipy.interpolate import spline
import numpy as np

import  runtime_parser as rp


def evaluate(probs, y_val, nClases,path,key):
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    y_val = y_val - 1
    for i in range(nClases):
        precision[i], recall[i], _ = precision_recall_curve(y_val == i, probs[:, i])
        average_precision[i] = average_precision_score(y_val == i, probs[:, i])

        plt.plot(recall[i], precision[i],
                 label='Kernel:{2} Precision-Recall curve of class{0} (area = {1:0.2f})' ''.format(i, average_precision[i],key))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")

        if rp.verbose:
            plt.draw()
            plt.pause(1)
        else:
            plt.savefig(path + "/Precision-Recall/" + key + " C " + str(rp.C) + " Gamma "  + str(rp.gamma) + " Degree "  +
                        str(rp.degree) + " Strategy " + str(rp.ovx) + ".png",bbox_inches='tight')

    plt.close()

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(nClases):
        fpr[i], tpr[i], _ = roc_curve(y_val == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    for i in range(nClases):

        """
        xnew = np.linspace(fpr[i].min(),fpr[i].max(),200)
        print fpr[i].shape
        print tpr[i].shape
        print "fpr",np.unique(fpr[i]).shape
        print "tpr", np.unique(tpr[i]).shape
        smooth = spline(np.unique(fpr[i]),np.unique(tpr[i]),xnew)
        """
        plt.plot(fpr[i], tpr[i], label='Kernel:{2} ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i],key))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="upper right")

        if rp.verbose:
            plt.draw()
            plt.pause(1)

        else:
            plt.savefig(path + "/ROC/" + key  + " C " + str(rp.C) + " Gamma "  + str(rp.gamma) + " Degree "  + str(rp.degree) + " Strategy " + str(rp.ovx) + ".png",bbox_inches='tight')

    plt.close()