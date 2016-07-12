from __future__ import division;
import numpy as np;
from matplotlib import pyplot as plt;
from matplotlib import colors
import matplotlib as mpl;
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import mlab;
from matplotlib import gridspec;
import pandas as pd
from IPython.display import display
import sklearn;
from sklearn.kernel_ridge import KernelRidge;

from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
              'theme': 'simple',
              'transition': 'none',
              'start_slideshow_at': 'selected',
});

def regression_example_draw(degree1, degree2, degree3, ifprint):
    x = np.linspace(0, 2*np.pi, 13);
    # np.random.randn generates gaussian samples
    y = np.sin(x) + np.random.randn(x.shape[0]) * 0.2;
    xx = np.linspace(0, 2*np.pi, 100);
    plt.figure(figsize=(12,7.5))
    plt.subplot(221)
    plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
    plt.plot(x, y, "or"); plt.hold(False)
    plt.legend(['True Curve','Data']); plt.title('Data and True Curve');

    # Here were are going to take advantage of numpy's 'polyfit' function
    # This implements a "polynomial fitting" algorithm
    # coeffs are the optimal coefficients of the polynomial
    coeffs = np.polyfit(x, y, degree1); # 0 is the degree of the poly
    # We construct poly(), the polynomial with "learned" coefficients
    poly = np.poly1d(coeffs);
    plt.subplot(222)
    plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
    plt.plot(x, y, "or");
    plt.plot(xx, poly(xx), color='b', linestyle='-'); plt.hold(False)
    plt.legend(['True Curve','Data','Learned Curve']); plt.title(str(degree1)+'th Order Polynomial')
    exprsn1=''
    for i in range(degree1+1):
        if i>0 and coeffs[i]>0:
            exprsn1 += '+%.3fx^%d' %(coeffs[i], i)
        elif i==0:
            exprsn1 += '%.3f' %(coeffs[i])
        else:
            exprsn1 += '%.3fx^%d' %(coeffs[i], i)

    coeffs = np.polyfit(x, y, degree2); # Now let's try degree = 1
    poly = np.poly1d(coeffs);
    plt.subplot(223)
    plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
    plt.plot(x, y, "or");
    plt.plot(xx, poly(xx), color='b', linestyle='-'); plt.hold(False)
    plt.legend(['True Curve','Data','Learned Curve']); plt.title(str(degree2)+'th Order Polynomial')
    exprsn2 = ''
    for i in range(degree2+1):
        if i>0 and coeffs[i]>0:
            exprsn2 += '+%.3fx^%d' %(coeffs[i], i)
        elif i==0:
            exprsn2 += '%.3f' %(coeffs[i])
        else:
            exprsn2 += '%.3fx^%d' %(coeffs[i], i)

    coeffs = np.polyfit(x, y, degree3); # Now degree = 3
    poly = np.poly1d(coeffs);
    plt.subplot(224)
    plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
    plt.plot(x, y, "or");
    plt.plot(xx, poly(xx), color='b', linestyle='-'); plt.hold(False)
    plt.legend(['True Curve','Data','Learned Curve']); plt.title(str(degree3)+'th Order Polynomial')
    plt.show()
    exprsn3 = ''
    for i in range(degree3+1):
        if i>0 and coeffs[i]>0:
            exprsn3 += '+%.3fx^%d' %(coeffs[i], i)
        elif i==0:
            exprsn3 += '%.3f' %(coeffs[i])
        else:
            exprsn3 += '%.3fx^%d' %(coeffs[i], i)

    if ifprint:
        print 'The expression for the first polynomial is y=' + exprsn1
        print 'The expression for the second polynomial is y=' + exprsn2
        print 'The expression for the third polynomial is y=' + exprsn3

def set_nice_plot_labels(axs):
    axs[0].set_title(r"$ \phi_j(x) = x^j$", fontsize=18, y=1.08);
    axs[0].set_xlabel("Polynomial", fontsize=18);
    axs[1].set_title(r"$ \phi_j(x) = \exp\left( - \frac{(x-\mu_j)^2}{2s^2} \right)$", fontsize=18, y=1.08);
    axs[1].set_xlabel("Gaussian", fontsize=18);
    axs[2].set_title(r"$ \phi_j(x) = (1  + \exp\left(\frac{\mu_j-x}{s}\right))^{-1}$", fontsize=18, y=1.08);
    axs[2].set_xlabel("Sigmoid", fontsize=18);

def basis_function_plot():
    x = np.linspace(-1,1,100);
    f, axs = plt.subplots(1, 3, sharex=True, figsize=(12,4));
    for j in range(8):
        axs[0].plot(x, np.power(x,j));
        axs[0].hold(True)
        axs[1].plot(x, np.exp( - (x - j/7 + 0.5)**2 / 2*5**2 ));
        axs[1].hold(True)
        axs[2].plot(x, 1 / (1 + np.exp( - (x - j/5 + 0.5) * 5)) );
        axs[2].hold(True)
    axs[0].hold(False)
    axs[1].hold(False)
    axs[2].hold(False)

    set_nice_plot_labels(axs) # I'm hiding some helper code that adds labels