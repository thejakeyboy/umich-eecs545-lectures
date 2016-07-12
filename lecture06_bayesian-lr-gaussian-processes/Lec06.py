from __future__ import division
# plotting
from matplotlib import pyplot as plt;
import matplotlib as mpl;
from mpl_toolkits.mplot3d import Axes3D
if "bmh" in plt.style.available: plt.style.use("bmh");
# matplotlib objects
from matplotlib import mlab;
from matplotlib import gridspec;
# scientific
import numpy as np;
import scipy as scp;
import scipy.stats;
# table display
import pandas as pd
from IPython.display import display
# scikit-learn
import sklearn;
from sklearn.kernel_ridge import KernelRidge;
# python
import random;
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


def plot_mvn(sigmax, sigmay, mux, muy, corr):
    # dimensions
    radius = 3 * max(sigmax, sigmay);
    # create grid
    x = np.linspace(mux-radius, mux+radius, 100);
    y = np.linspace(muy-radius, muy+radius, 100);
    X, Y = np.meshgrid(x, y);

    # data limits
    xlim = (x.min(), x.max());
    ylim = (y.min(), y.max());

    # bivariate and univariate normals
    sigmaxy = corr * np.sqrt(sigmax * sigmay);
    Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mux, muy, sigmaxy);
    zx = np.sum(Z, axis=0); #mlab.normpdf(x, mux, sigmax);
    zy = np.sum(Z, axis=1); #mlab.normpdf(y, muy, sigmay);

    # figure
    fig = plt.figure(figsize=(6,6));

    # subplots
    gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5]);
    ax_xy = fig.add_subplot(gs[1,0]);
    ax_xy.set_xlabel('$x_a$',fontsize=20)
    ax_xy.set_ylabel('$x_b$',rotation=0,fontsize=20)
    ax_xy.set_title('$P(x_a, x_b)$',fontsize=20)
    ax_x  = fig.add_subplot(gs[0,0], sharex=ax_xy);
    ax_x.set_xlabel('$x_a$')
    ax_x.set_title('$P(x_a)$',fontsize=20)
    ax_y  = fig.add_subplot(gs[1,1], sharey=ax_xy);
    ax_y.set_title('$P(x_b)$',fontsize=20)
    ax_y.set_ylabel('$x_b$')

    # plot
    ax_xy.imshow(Z, origin='lower', extent=xlim+ylim, aspect='auto');
    ax_x.plot(x, zx);
    ax_y.plot(zy, y);

    # hide labels
    ax_x.xaxis.set_visible(False);
    ax_x.yaxis.set_visible(False);
    ax_y.xaxis.set_visible(False);
    ax_y.yaxis.set_visible(False);

    # layout & title
    plt.tight_layout();

