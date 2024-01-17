from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from K_MEAN import K_mean
import numpy as np

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)
plt.scatter(X[:,0],X[:,1],c= "black",marker= ".")
plt.show()
nowitstime= K_mean(no_iters=5)
nowitstime._points(X)