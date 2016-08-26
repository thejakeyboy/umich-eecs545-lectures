
from __future__ import division
# plotting
from matplotlib import pyplot as plt;
import matplotlib as mpl;
from matplotlib import gridspec;
from mpl_toolkits.mplot3d import Axes3D
if "bmh" in plt.style.available: plt.style.use("bmh");
    
# scientific
import numpy as np;
import scipy as scp;
import scipy.stats;

# scikit-learn
import sklearn;
from sklearn.kernel_ridge import KernelRidge;

# python
import random;

from IPython.display import Image
from IPython.core.display import HTML 

# warnings
import warnings
warnings.filterwarnings("ignore")

# rise config
from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
              'theme': 'simple',
              'start_slideshow_at': 'selected',
              'transition':'fade',
              'scroll': False
});


# ADAPTED FROM SCIKIT-LEARN EXAMPLE
# (http://scikit-learn.org/stable/modules/svm.html)
def max_margin_classifier(fig, X, Y):
    ax = fig.add_subplot(122, adjustable="box-forced", aspect='equal')
    clf = sklearn.svm.SVC(kernel='linear')
    clf.fit(X, Y)

    # get the separating hyperplane
    w = clf.coef_[0]
    m = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = m * xx - (clf.intercept_[0]) / w[1]

    # plot the parallel lines to the separating hyperplane that pass through the
    # support vectors
    t = clf.support_vectors_[0]
    yy_down = m * xx + (t[1] - m * t[0])
    t = clf.support_vectors_[-1]
    yy_up = m * xx + (t[1] - m * t[0])

    ax.plot(xx, yy, 'g-')
    ax.plot(xx, yy_down, 'g--')
    ax.plot(xx, yy_up, 'g--')

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=120, facecolors='none', edgecolors='k', linewidth=3);
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, s=40)
    plt.xlim([-5, 5]); plt.ylim([-5, 5]);
    ax.quiver(0, -clf.intercept_[0]/ w[1], w[0]/np.sqrt(w[0]**2+w[1]**2), w[1]/np.sqrt(w[0]**2+w[1]**2), angles='xy', scale_units='xy', scale=0.75, color='r')
    plt.title("Maximum Margin Classifier");


    ax = fig.add_subplot(121, adjustable="box-forced", aspect='equal')
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, s=40)
    xx = np.linspace(-5, 5)

    # Maximum Margin Classifier
    line1, = ax.plot(xx, yy, 'g')
    sv = clf.support_vectors_[[0, -1], :]
    proj_x = ( sv[:,0]+m*sv[:,1]-m*(- clf.intercept_[0]/w[1]) )/(m**2+1)
    proj_y = proj_x*m - clf.intercept_[0]/w[1]
    proj = np.array([proj_x, proj_y])
    proj = proj.T
    ax.plot(np.array([sv[0,0], proj[0,0]]), np.array([sv[0,1], proj[0,1]]), 'g--')

    # Non-maximum Margin Classifier
    X_new = np.concatenate((X[0:30,:], np.array([[1, -4]]), X[30:60,:]))
    Y_new = [0] * 30 + [1] * 31
    clf = sklearn.svm.SVC(kernel='linear')
    clf.fit(X_new, Y_new)
    w = clf.coef_[0]
    m = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = m * xx - (clf.intercept_[0]) / w[1]
    line2, = ax.plot(xx, yy, 'b')
    sv = clf.support_vectors_[[0, -1], :]
    proj_x = ( sv[:,0]+m*sv[:,1]-m*(- clf.intercept_[0]/w[1]) )/(m**2+1)
    proj_y = proj_x*m - clf.intercept_[0]/w[1]
    proj = np.array([proj_x, proj_y])
    proj = proj.T
    ax.plot(np.array([sv[0,0], proj[0,0]]), np.array([sv[0,1], proj[0,1]]), 'b--')

    # Misclassify
    X_new = X[30:60, :]
    Y_new = [0] * 15 + [1] * 15
    clf = sklearn.svm.SVC(kernel='linear')
    clf.fit(X_new, Y_new)
    w = clf.coef_[0]
    m = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = m * xx - (clf.intercept_[0]) / w[1]
    line3, = ax.plot(xx, yy, 'r')
    plt.xlim([-5, 5]); plt.ylim([-5, 5]);

    plt.legend((line1, line2, line3), ('Max Margin', 'Non-max Margin', 'Misclassify'), fontsize='small')
    plt.title("Different Classifiers");

def plot_svc():
    # Create 30 random points
    np.random.seed()
    X = np.r_[np.random.randn(30, 2) - [3, 3], np.random.randn(30, 2) + [3, 3]]
    Y = [0] * 30 + [1] * 30

    fig = plt.figure(figsize=(11,4))
    max_margin_classifier(fig, X, Y)