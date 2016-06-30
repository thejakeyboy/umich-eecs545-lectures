from __future__ import division;
import numpy as np;
from matplotlib import pyplot as plt;

def plot_optimization(ax, fn, d1, lst, xlim=None, ylim=None, title="", tangents=False):
    # Determine Domain
    minX = min(lst); maxX = max(lst);
    rng = maxX-minX;
    minX -= rng/3; maxX += rng/3;
    
    if xlim is None: xvals = np.linspace(minX, maxX, 100);
    else:            xvals = np.linspace(xlim[0], xlim[1], 100);
    
    # evaluate
    yvals = fn(xvals);
    
    # axes
    plt.plot([ xvals[0], xvals[-1] ], [0, 0], "k-", linewidth=1, alpha=0.5);
    plt.plot([ 0, 0 ], [ min(yvals), max(yvals)], "k-", linewidth=1, alpha=0.5);
    
    # Plot Function
    ax.plot(xvals, yvals);
    
    # Plot Tangents
    if tangents:
        for x in lst[:-1]:
            deriv = d1(x);
            xx = [xvals[0], xvals[-1]];
            yy = [fn(x) - deriv * (x - xvals[0]), fn(x) + deriv * (xvals[-1] - x)]
            ax.plot(xx, yy, "--r", linewidth=1);
        
    # Plot Iterates
    for x0,x1 in zip(lst[:-1],lst[1:]):
        xx, yy = [x0, x1], [fn(x0), fn(x1)];
        #ax.plot(xx, yy, "or");
        ax.plot(xx, yy, "-r");

    # Plot Markers
    for i in range(0,lst.__len__()):
        x = lst[i]
        xx = [x, x];
        yy = [fn(x), 0];
        ax.plot(xx, yy, "--c");
        ax.plot(xx, yy, "oc");
        ax.text(x, 0, "x_"+str(i+1), fontsize=12, horizontalalignment='center', verticalalignment='top')

    if xlim is not None: ax.set_xlim(xlim);
    if ylim is not None: ax.set_ylim(ylim);
    
    return ax;

def newton_exact(d1, d2, x0, eps=1e-10, maxn=1e2, lst=None):
    """
    Runs Newton's method using exact analytic derivatives.
    @param (d1: R --> R) Analytic first derivative.
    @param (d2: R --> R) Analytic second derivative 
    """
    for n in range(int(maxn)):
        if lst is not None: lst.append(x0);
        x1 = x0-d1(x0)/d2(x0);
        if abs(x1-x0) < eps: return x0; 
        x0 = x1;
    print("Newton's Method did not converge.");
    return x0;

def newton_exact_ver2(fn, d1, x0, eps=1e-10, maxn=1e2, lst=None):
    """
    Runs Newton's method using exact analytic derivatives.
    @param (fn: R --> R) Analytic f(x).
    @param (d1: R --> R) Analytic first derivative
    """
    for n in range(int(maxn)):
        if lst is not None: lst.append(x0);
        x1 = x0-fn(x0)/d1(x0);
        if abs(x1-x0) < eps: return x0;
        x0 = x1;
    return lst;