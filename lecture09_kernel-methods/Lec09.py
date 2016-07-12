from __future__ import division

# plotting
from matplotlib import pyplot as plt;
from matplotlib import colors
import matplotlib as mpl;
from mpl_toolkits.mplot3d import Axes3D
if "bmh" in plt.style.available: plt.style.use("bmh");

# matplotlib objects
from matplotlib import mlab;
from matplotlib import gridspec;

# scientific
import numpy as np;
import scipy as scp;
from scipy import linalg
import scipy.stats;

# table display
import pandas as pd
from IPython.display import display

# python
import random;

# scikit-learn
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge

# warnings
import warnings
warnings.filterwarnings("ignore")

from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
              'theme': 'simple',
              'start_slideshow_at': 'selected',
              'transition':'fade',
              'scroll': False
});

def get_circular_data(n):
    # sample outer circle
    theta0 = np.random.random(n) * 2 * np.pi;
    r0 = 2 + np.random.randn(n) / 3;
    # sample inner circle
    theta1 = np.random.random(n) * 2 * np.pi;
    r1 = 5 + np.random.randn(n) / 3;
    # join data
    x0 = r0 * [np.cos(theta0), np.sin(theta0)];
    x1 = r1 * [np.cos(theta1), np.sin(theta1)];
    return x0, x1;

def plot_circular_data(n):
    # get data
    x0, x1 = get_circular_data(n);

    # plot
    plt.figure(figsize=(8,5))
    plt.scatter(*x0, c="r", marker="o", s=50)
    plt.scatter(*x1, c="b", marker="^", s=50)
    plt.axis("equal");

def plot_circular_3d(n):
    # get data
    x0, x1 = get_circular_data(n);

    # plot
    fig = plt.figure(figsize=(7,5));
    ax = fig.add_subplot(111, projection='3d');
    # plot 3d points
    ax.scatter(x0[0],x0[1], np.linalg.norm(x0, axis=0)**2, c="r", s=50)
    ax.scatter(x1[0], x1[1], np.linalg.norm(x1, axis=0)**2, c="b", marker="^", s=50);
    # set camera
    ax.view_init(elev=20, azim=45);

def plot_circular_squared(n):
    # get data
    x0, x1 = get_circular_data(n);

    # plot
    plt.figure(figsize=(5,7))
    plt.scatter(*(x0**2), c="r", s=50)
    plt.scatter(*(x1**2), c="b", marker="^", s=50)
    plt.axis("equal");

def plot_solution():
    # get data
    x0, x1 = get_circular_data(100);

    # plot
    fig = plt.figure(figsize=(15,5));
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax = fig.add_subplot(gs[0], projection='3d');
    # plot 3d points
    ax.scatter(x0[0],x0[1], np.linalg.norm(x0, axis=0)**2, c="r", s=50)
    ax.scatter(x1[0], x1[1], np.linalg.norm(x1, axis=0)**2, c="b", marker="^", s=50);
    # set camera
    ax.view_init(elev=20, azim=45);

    ax = fig.add_subplot(gs[1]);
    ax.scatter(*(x0**2), c="r", s=50)
    ax.scatter(*(x1**2), c="b", marker="^", s=50)
    ax.axis("equal");

def poly_reg_example():
    x = np.random.random(50) * 2*np.pi - np.pi;
    y = np.sin(x);
    # format into matrix
    X = x[:, np.newaxis];

    # plot ground truth
    plt.figure(figsize=(8,5))
    x_plot = np.linspace(-np.pi, np.pi,100)[:, np.newaxis];
    plt.plot(x_plot, np.sin(x_plot), label="ground truth")
    plt.scatter(x, y, label="data-points")

    # linear regression solely with x
    w = np.dot(np.linalg.pinv(np.array([x]).T),y)
    plt.plot(x_plot, w*x_plot, label="No Feature Mapping")

    # polynomial regression for degree 5
    for degree in [5]:
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(x_plot)
        plt.plot(x_plot, y_plot, label="Degree %d Polynomial Mapping" % degree)

    plt.legend(loc='lower right')
    plt.show()

def plot_kernel_ridge(X, y, gamma=0.5, alpha=0.1):
    # kernel (ridge) regression
    krr = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha);
    krr.fit(X,y);

    # predict
    x_plot = np.linspace(min(X), max(X), 100)[:,np.newaxis];
    y_plot = krr.predict(x_plot);

    # plot
    plt.figure(figsize=(8,4.8));
    plt.plot(X, y, 'or');
    plt.plot(x_plot, y_plot)
#     plt.title(r"Gaussian Kernel ($\gamma=%0.2f, \alpha=%0.2f$)" % (gamma,alpha), fontsize=16)
    plt.title(r"Gaussian Kernel ($\gamma=%0.2f$)" % (gamma), fontsize=16)
