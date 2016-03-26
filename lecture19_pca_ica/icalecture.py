from __future__ import division
import numpy as np
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
import seaborn

# original samples
S = np.random.rand(2,500)
plt.subplot(2,2,1)
plt.scatter(S[0,:], S[1,:])
plt.title('Original')

# observed samples
A = np.array([[1, 2], [-2, 1]])
X = A.dot(S)
plt.subplot(2,2,2)
plt.scatter(X[0,:], X[1,:])
plt.title('Mixed (observed)')

# PCA recovered samples
_,_,V = np.linalg.svd(X)
plt.subplot(2,2,3)
plt.scatter(V[0,:], V[1,:])
plt.title('PCA')

# ICA recovered samples
fica = FastICA()
Y = fica.fit_transform(X.T).T
plt.subplot(2,2,4)
plt.scatter(Y[0,:], Y[1,:])
plt.title('ICA')


plt.show()


