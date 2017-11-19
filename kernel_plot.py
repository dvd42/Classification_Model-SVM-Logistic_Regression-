from sklearn import svm, datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import runtime_parser as rp

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

   Parameters
   ----------
   x: data to base x-axis meshgrid on
   y: data to base y-axis meshgrid on
   h: stepsize for meshgrid, optional

   Returns
   -------
   xx, yy : ndarray
   """

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

   Parameters
   ----------
   ax: matplotlib axes object
   clf: a classifier
   xx: meshgrid ndarray
   yy: meshgrid ndarray
   params: dictionary of params to pass to contourf, optional
   """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_kernel(models, x_train, y_train,path):

    pca = PCA(n_components=2)
    x_train = pca.fit_transform(x_train)

    trained_models = (clf.fit(x_train, y_train) for clf in models)

    # title for the plots
    titles = ('Rbf kernel (gamma %.2f)' %rp.gamma,
            'Poly kernel (degree %d)' %rp.degree,
            'Sigmoid kernel',
            'Linear kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = x_train[:, 0], x_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(trained_models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
              cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        #ax.set_xlabel()
        #ax.set_ylabel()
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    if rp.verbose:
        plt.draw()
        plt.pause(5)
    else:
        plt.savefig(path + "/Kernels/" + " C " + str(rp.C) + " Gamma "  + str(rp.gamma) + " Degree "  +
                        str(rp.degree) + " Strategy " + str(rp.ovx) + ".png",bbox_inches='tight')


    plt.close()