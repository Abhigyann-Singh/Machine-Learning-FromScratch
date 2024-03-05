import numpy as np

class NaivesBayes():
    
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classses = len(self._classes)
        
        #calculate mean, variance, and prior for each class
        self._mean = np.zeros((n_classses,n_features),dtype=np.float64)
        self._var = np.zeros((n_classses,n_features),dtype=np.float64)
        self._priors = np.zeros(n_classses,dtype=np.float64)
        
        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
    
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self,x):
        prosteriors = []
        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            prosterior = np.sum(np.log(self._pdf(idx,x)))
            prosterior = prior + prosterior
            prosteriors.append(prosterior)
        #return class with highest posterior probability
        return self._classes[np.argmax(prosteriors)]
    
    def _pdf(self,class_idx,x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp(- (x-mean)**2 / (2 * var))
        den = np.sqrt(2 * np.pi * var)
        return num/den
            