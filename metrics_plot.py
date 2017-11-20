from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from matplotlib import pyplot as plt
import  runtime_parser as rp

import file_writer as fw


def evaluate(model, x_test, y_test, nClases, path, key):
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()


    probs = model.predict_proba(x_test)
    y_pred = model.predict(x_test)

    for i in range(nClases):
        precision[i], recall[i], _ = precision_recall_curve(y_test == i, probs[:, i])
        average_precision[i] = average_precision_score(y_test == i, probs[:, i])


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
        fpr[i], tpr[i], _ = roc_curve(y_test == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()


    micro = f1_score(y_test,y_pred,average="micro")
    macro = f1_score(y_test,y_pred,average="macro")
    #weighted = f1_score(y_test,y_pred,average="weighted")

    fw.store_fscore(key,micro,macro,path)



    for i in range(nClases):

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