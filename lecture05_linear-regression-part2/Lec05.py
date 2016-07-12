from __future__ import division
# plotting
from matplotlib import pyplot as plt;
if "bmh" in plt.style.available: plt.style.use("bmh");
# scientific
import numpy as np;
import pandas as pd
import scipy as scp;
import scipy.stats;
# Nice Plot of Pandas DataFrame
from IPython.display import display, HTML
# rise config
from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
              'theme': 'simple',
              'start_slideshow_at': 'selected',
})
import warnings
warnings.simplefilter("ignore")

def regression_overfitting_degree(degree0, degree1, degree2, degree3):
    degreelist = np.array([degree0, degree1, degree2, degree3]);
    x = np.linspace(0, 2*np.pi, 13);
    # np.random.randn generates gaussian samples
    y = np.sin(x) + np.random.randn(x.shape[0]) * 0.3;
    xx = np.linspace(0, 2*np.pi, 100);
    plt.figure(figsize=(12,7.5)) ;
    for i in range(0,4):
        degree = degreelist[i]
        coeffs = np.polyfit(x, y, degree);
        poly = np.poly1d(coeffs);
        plt.subplot(2,2,i+1);
        plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
        plt.plot(x, y, "or");
        plt.plot(xx, poly(xx), color='b', linestyle='-'); plt.hold(False)
        plt.legend(['True Curve','Data','Learned Curve']);
        plt.title(str(degree)+'th Order Polynomial')

def regression_overfitting_datasetsize(size0, size1, size2, size3):
    sizelist = np.array([size0, size1, size2, size3]);
    degree = 12;
    plt.figure(figsize=(12,7.5)) ;
    for i in range(0,4):
        size = sizelist[i]
        x = np.linspace(0, 2*np.pi, size);
        y = np.sin(x) + np.random.randn(x.shape[0]) * 0.3;
        xx = np.linspace(0, 2*np.pi, 100);
        coeffs = np.polyfit(x, y, degree);
        poly = np.poly1d(coeffs);
        plt.subplot(2,2,i+1);
        plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
        plt.plot(x, y, "or");
        plt.plot(xx, poly(xx), color='b', linestyle='-'); plt.hold(False)
        plt.legend(['True Curve','Data','Learned Curve']);
        plt.title('Training Dataset Size = ' + str(size))

def regression_overfitting_curve():
    #   Plot the training error and test error w.r.t. dataset size
    numItrn = 100
    degreelist = range(0, 14)
    numTrain = 20
    numTest = 50
    dataTrain = np.linspace(0, 2*np.pi, numTrain)
    labelTrain = np.sin(dataTrain) + np.random.randn(dataTrain.shape[0]) * 0.3
    dataTest = np.linspace(0, 2*np.pi, numTest)
    labelTest = np.sin(dataTest) + np.random.randn(dataTest.shape[0]) * 0.3
    errTrain = np.zeros(degreelist.__len__())
    errTest = np.zeros(degreelist.__len__())
    for j in range(0, numItrn):
        for i in range(0, degreelist.__len__()):
            degree = degreelist[i]
            coeffs = np.polyfit(dataTrain, labelTrain, degree)
            poly = np.poly1d(coeffs)
            predTrain = poly(dataTrain)
            predTest = poly(dataTest)
            errTrain[i] += np.sqrt(((predTrain - labelTrain)**2).sum()/numTrain)/numItrn
            errTest[i] += np.sqrt(((predTest - labelTest)**2).sum()/numTest)/numItrn
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(degreelist, errTrain, color='r', linestyle='-', marker='o'); plt.hold(True)
    plt.plot(degreelist, errTest, color='b', linestyle='-', marker='x'); plt.hold(False)
    plt.legend(['Training Error','Test Error'])
    plt.xlabel('Degree of Linear Regression'); plt.ylabel('RMSE');
    plt.title('Training Error and Test Error v.s. Degree')
    #   Plot the training error and test error w.r.t. degree
    sizelist = 10 * np.array(range(1, 16));
    numTest = 50
    degree = 12
    errTrain = np.zeros(sizelist.shape[0])
    errTest = np.zeros(sizelist.shape[0])
    for j in range(0, numItrn):
        for i in range(0, sizelist.shape[0]):
            numTrain = sizelist[i]
            dataTrain = np.linspace(0, 2*np.pi, numTrain)
            labelTrain = np.sin(dataTrain) + np.random.randn(dataTrain.shape[0]) * 0.3
            dataTest = np.linspace(0, 2*np.pi, numTest)
            labelTest = np.sin(dataTest) + np.random.randn(dataTest.shape[0]) * 0.3
            coeffs = np.polyfit(dataTrain, labelTrain, degree)
            poly = np.poly1d(coeffs)
            predTrain = poly(dataTrain)
            predTest = poly(dataTest)
            errTrain[i] += np.sqrt(((predTrain - labelTrain)**2).sum()/numTrain)/numItrn
            errTest[i] += np.sqrt(((predTest - labelTest)**2).sum()/numTest)/numItrn
    plt.subplot(1,2,2)
    plt.plot(sizelist, errTrain, color='r', linestyle='-', marker='o'); plt.hold(True)
    plt.plot(sizelist, errTest, color='b', linestyle='-', marker='x'); plt.hold(False)
    plt.legend(['Training Error','Test Error'])
    plt.xlabel('Training Dataset Size'); plt.ylabel('RMSE');
    plt.title('Training Error and Test Error v.s. Training Dataset Size')

def regression_overfitting_coeffs():
    degree0 = 0
    degree1 = 3
    degree2 = 9
    degree3 = 12
    degreelist = np.array([degree0, degree1, degree2, degree3])
    x = np.linspace(0, 2*np.pi, degree3+1)
    # np.random.randn generates gaussian samples
    y = np.sin(x) + np.random.randn(x.shape[0]) * 0.3
    y = 100*y
    coeffs = np.zeros([4,degree3+1])
    for i in range(0,4):
        degree = degreelist[i]
        coeffs[i,0:degree+1] = np.polyfit(x, y, degree)
    coeffs[coeffs == 0] = float('nan')
    df = pd.DataFrame(
        coeffs.T,
        columns=["M=0 (Underfitting)", "M=3 (Good)", "M=9 (Overfitting)","M=12 (Overfitting)"],
        index=["w_0","w_1","w_2","w_3","w_4","w_5","w_6","w_7","w_8","w_9","w_10","w_11","w_12"]
    )
    display(df.fillna(''));

def regression_regularization(x, y, xx, degree, lamda):
    Phi_x = np.zeros([x.__len__(), degree+1])
    for i in range(0, degree + 1):
        Phi_x[:, i] = x**i
    coeffs = np.dot(np.dot(np.linalg.inv(np.dot(Phi_x.T,Phi_x) + lamda*np.eye(degree+1)), Phi_x.T), y)
    Phi_xx = np.zeros([xx.__len__(), degree+1])
    for i in range(0, degree + 1):
        Phi_xx[:, i] = xx**i
    prediction_x = np.dot(Phi_x, coeffs)
    prediction_xx = np.dot(Phi_xx, coeffs)
    return prediction_xx, prediction_x, coeffs

def regression_regularization_plot():
    degree = 10
    numData = 13
    x = np.linspace(0, 2*np.pi, numData)
    # np.random.randn generates gaussian samples
    y = np.sin(x) + np.random.randn(x.shape[0]) * 0.4
    xx = np.linspace(0, 2*np.pi, 100)

    plt.figure(figsize=(12,7.5))

    # Plot the Ordinary Least Squares, lambda=0
    plt.subplot(2,2,1)
    prediction_xx, _, _ = regression_regularization(x, y, xx, degree, lamda=0)
    plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
    plt.plot(x, y, "or")
    plt.plot(xx, prediction_xx, color='b', linestyle='-'); plt.hold(False)
    plt.legend(['True Curve','Data','Learned Curve'])
    plt.title('Ordinary Least Squares (Degree='+str(degree)+')')

    # Plot L2 Regularized Least Squares, lambda=e^1
    plt.subplot(2,2,2)
    prediction_xx, _, _ = regression_regularization(x, y, xx, degree, lamda=np.exp(-1))
    plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
    plt.plot(x, y, "or")
    plt.plot(xx, prediction_xx, color='b', linestyle='-'); plt.hold(False)
    plt.legend(['True Curve','Data','Learned Curve'])
    plt.title('L2 Regularization (Degree=' + str(degree) + ', $\lambda$=$e^{-1}$' + ')')

    # Plot L2 Regularized Least Squares, lambda=e^40
    plt.subplot(2,2,3)
    prediction_xx, _, _ = regression_regularization(x, y, xx, degree, lamda=np.exp(30))
    plt.plot(xx, np.sin(xx), "g", linestyle='--'); plt.hold(True)
    plt.plot(x, y, "or")
    plt.plot(xx, prediction_xx, color='b', linestyle='-'); plt.hold(False)
    plt.legend(['True Curve','Data','Learned Curve'])
    plt.title('L2 Regularization (Degree=' + str(degree) + ', $\lambda$=$e^{30}$' + ')')

    # Plot the Training Error and Test Error vs. Regularization Coefficient
    plt.subplot(2,2,4)
    lamdalist = np.logspace(-20,30,50, base=np.e)
    numTrain = 15
    numTest = 50
    numItrn = 100
    degree = 10
    errTrain = np.zeros(lamdalist.shape[0])
    errTest = np.zeros(lamdalist.shape[0])
    for j in range(0, numItrn):
        dataTrain = np.linspace(0, 2*np.pi, numTrain)
        labelTrain = np.sin(dataTrain) + np.random.randn(dataTrain.shape[0]) * 0.4
        dataTest = np.linspace(0, 2*np.pi, numTest)
        labelTest = np.sin(dataTest) + np.random.randn(dataTest.shape[0]) * 0.4
        for i in range(0, lamdalist.shape[0]):
            lamda = lamdalist[i]
            predTest, predTrain, _ = regression_regularization(dataTrain, labelTrain, dataTest, degree, lamda)
            errTrain[i] += np.sqrt(((predTrain - labelTrain)**2).sum()/numTrain)/numItrn
            errTest[i] += np.sqrt(((predTest - labelTest)**2).sum()/numTest)/numItrn
    plt.plot(np.log(lamdalist), errTrain, color='r', linestyle='-', marker='o'); plt.hold(True)
    plt.plot(np.log(lamdalist), errTest, color='b', linestyle='-', marker='x'); plt.hold(False)
    plt.legend(['Training Error','Test Error'], loc='lower right')
    plt.xlabel('ln($\lambda$)'); plt.ylabel('RMSE')
    plt.title('Training Error and Test Error v.s. $\lambda$')

def regression_regularization(x, y, xx, degree, lamda):
    Phi_x = np.zeros([x.__len__(), degree+1])
    for i in range(0, degree + 1):
        Phi_x[:, i] = x**i
    coeffs = np.dot(np.dot(np.linalg.inv(np.dot(Phi_x.T,Phi_x) + lamda*np.eye(degree+1)), Phi_x.T), y)
    Phi_xx = np.zeros([xx.__len__(), degree+1])
    for i in range(0, degree + 1):
        Phi_xx[:, i] = xx**i
    prediction_x = np.dot(Phi_x, coeffs)
    prediction_xx = np.dot(Phi_xx, coeffs)
    return prediction_xx, prediction_x, coeffs

def regression_regularization_coeff():
    lamdalist = np.array([0, np.exp(1), np.exp(10)])
    degree = 10
    x = np.linspace(0, 2*np.pi, degree+1)
    xx = np.linspace(0, 2*np.pi, degree+1)
    y = np.sin(x) + np.random.randn(x.shape[0]) * 0.3
    y = 100*y

    coeffs = np.zeros([3,degree+1])
    for i in range(0,lamdalist.__len__()):
        lamda = lamdalist[i]
        _, _, coeffs[i,0:degree+1] = regression_regularization(x, y, xx, degree, lamda)

    coeffs[coeffs == 0] = float('nan')
    df = pd.DataFrame(
        coeffs.T,
        columns=["lambda=0", "lambda=exp^1", "lambda=exp^10"],
        index=["w_0","w_1","w_2","w_3","w_4","w_5","w_6","w_7","w_8","w_9","w_10"]
    )
    display(df.fillna(''));