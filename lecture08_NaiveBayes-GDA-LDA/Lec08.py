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

def lin_reg_classifier(means, covs, n, outliers):
    """
    Least Squares for Classification.

    :Parameters:
      - `means`: means of multivariate normal distributions used to generate data.
      - `covs`: terms of variance-covariance matrix used to determine spread of simulated data.
      - `n`: number of samples.
      - `outliers`: user-specified outliers to be added to the second simulated dataset.
    """
    # generate data
    x1, y1 = np.random.multivariate_normal(means[0], covs[0], n[0]).T
    x2, y2 = np.random.multivariate_normal(means[1], covs[1], n[1]).T
    # add targets
    class_1 = [1]*n[0] + [0]*n[1]
    class_2 = [0]*n[0] + [1]*n[1]
    T = np.mat([class_1, class_2]).T
    # add intercept and merge data
    ones = np.ones(n[0]+n[1])
    a = np.hstack((x1,x2))
    b = np.hstack((y1,y2))
    X = np.mat([ones, a, b]).T
    # obtain weights
    w_t = np.dot(T.T, np.linalg.pinv(X).T)
    # obtain decision line
    decision_line_int = -(w_t.item((0,0)) - w_t.item((1,0)))/(w_t.item((0,2)) - w_t.item((1,2)))
    decision_line_slope = - (w_t.item((0,1)) - w_t.item((1,1)))/(w_t.item((0,2)) - w_t.item((1,2)))
    # add outliers to the second set of simulated data
    extract_x = []
    extract_y = []
    for i in outliers:
        extract_x.append(i[0])
        extract_y.append(i[1])
    x2_out = np.hstack((x2, extract_x))
    y2_out = np.hstack((y2, extract_y))
    class_1_out = [1]*n[0] + [0]*n[1] + [0]*len(outliers)
    class_2_out = [0]*n[0] + [1]*n[1] + [1]*len(outliers)
    T_out = np.array([class_1_out, class_2_out]).T
    ones_out = np.ones(n[0]+n[1]+len(outliers))
    a_out = np.hstack((x1,x2_out))
    b_out = np.hstack((y1,y2_out))
    X_out = np.array([ones_out, a_out, b_out]).T
    # obtain revised weights and decision line
    w_t_out = np.dot(T_out.T, np.linalg.pinv(X_out).T)
    decision_line_int_out = -(w_t_out[0][0] - w_t_out[1][0])/(w_t_out[0][2] - w_t_out[1][2])
    decision_line_slope_out = - (w_t_out[0][1] - w_t_out[1][1])/(w_t_out[0][2] - w_t_out[1][2])

    # plot results
    x = np.linspace(np.min(a_out)-3 , np.max(a_out)+3, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)
    plt.suptitle('Least Squares for Classification')
    ax1.plot(x, decision_line_int+decision_line_slope*x, 'k', linewidth=2)
    ax1.plot(x1, y1, 'go', x2, y2, 'bs', alpha=0.4)
    ax2.plot(x, decision_line_int_out+decision_line_slope_out*x, 'k', linewidth=2)
    ax2.plot(x1, y1, 'go', x2, y2, 'bs', alpha=0.4)
    for i in range(len(outliers)):
        ax2.plot(outliers[i][0], outliers[i][1], 'bs', alpha=0.4)
    fig.set_size_inches(15, 5, forward=True)
    ax1.set_xlim([np.min(a_out)-1, np.max(a_out)+1,])
    ax2.set_xlim([np.min(a_out)-1, np.max(a_out)+1])
    ax1.set_ylim([np.min(b_out)-1, np.max(b_out)+1,])
    ax2.set_ylim([np.min(b_out)-1, np.max(b_out)+1])
    ax1.set_xlabel('X1')
    ax2.set_xlabel('X1')
    ax1.set_ylabel('X2')
    plt.show()

def generate_gda(means, covs, num_samples):
    num_classes = len(means);
    num_samples //= num_classes;

    # cheat and draw equal number of samples from each gaussian
    samples = [
        np.random.multivariate_normal(means[c],covs[c],num_samples).T
        for c in range(num_classes)
    ];

    return np.concatenate(samples, axis=1);

def plot_decision_contours(means, covs):
    # plt
    fig = plt.figure(figsize=(10,6));
    ax = fig.gca();

    # generate samples
    data_x,data_y = generate_gda(means, covs, 1000);
    ax.plot(data_x, data_y, 'x');

    # dimensions
    min_x, max_x = -10,10;
    min_y, max_y = -10,10;

    # grid
    delta = 0.025
    x = np.arange(min_x, max_x, delta);
    y = np.arange(min_y, max_y, delta);
    X, Y = np.meshgrid(x, y);

    # bivariate difference of gaussians
    mu1,mu2 = means;
    sigma1, sigma2 = covs;
    Z1 = mlab.bivariate_normal(X, Y, sigmax=sigma1[0][0], sigmay=sigma1[1][1], mux=mu1[0], muy=mu1[1], sigmaxy=sigma1[0][1]);
    Z2 = mlab.bivariate_normal(X, Y, sigmax=sigma2[0][0], sigmay=sigma2[1][1], mux=mu2[0], muy=mu2[1], sigmaxy=sigma2[0][1]);
    Z = Z2 - Z1;

    # contour plot
    ax.contour(X, Y, Z, levels=np.linspace(np.min(Z),np.max(Z),10));
    cs = ax.contour(X, Y, Z, levels=[0], c="k", linewidths=5);
    plt.clabel(cs, fontsize=10, inline=1, fmt='%1.3f')

    # plot settings
    ax.set_xlim((min_x,max_x));
    ax.set_ylim((min_y,max_y));

#     ax.set_title("Gaussian Discriminant Analysis:  $P(y=1 | x) - P(y=0 | x)$", fontsize=20)
    ax.set_title("Countours:  $P(y=1 | x) - P(y=0 | x)$", fontsize=20)