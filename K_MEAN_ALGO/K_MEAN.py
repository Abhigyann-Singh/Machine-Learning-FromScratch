import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance (x1,x2):
    return np.sqrt(np.sum((x2-x1)**2))


class K_mean:
    def __init__(self, no_iters = 4 , no_centroid= 3):
        self.no_iters= no_iters
        self.no_centroid= no_centroid
        self.centroid = []
        
    def _points (self,X):
        self.no_test= X.shape[0]
        self.cent_assigned = np.zeros(self.no_test)
        cent_idxs= np.random.choice(self.no_test, self.no_centroid , replace = False)
        self.centroid = X[cent_idxs,:]
        
        
        for _ in range(self.no_iters):
            plt.scatter(self.centroid[:,0],self.centroid[:,1],c="blue")
            plt.show()
            self._assign_centres(X)     
            self._plot_colour(X)
            self._calculate_mean(X)
    
    def _assign_centres (self, X):
        for j in range(self.no_test):
                shor_dis = 100000000
                for i in range(self.no_centroid):
                    _dis = euclidean_distance(self.centroid[i,:],X[j,:])
                    if _dis<shor_dis:
                        shor_dis=_dis
                        self.cent_assigned[j] = i
            
    
    def _calculate_mean(self,X):
        for i in range(self.no_centroid):
            indxxx =np.where(self.cent_assigned == i)[0]
            self.centroid[i,0]= (1/len(indxxx))*np.sum(X[indxxx,0])
            self.centroid[i,1]= (1/len(indxxx))*np.sum(X[indxxx,1])
        
        
        
    def _plot_colour (self,X):
        colour = ["blue","yellow","red"]
        for i in range(self.no_centroid):
            indxxx =np.where(self.cent_assigned == i)[0]
            plt.scatter(X[indxxx,0],X[indxxx,1],marker=".",c=colour[i])
        plt.show()    
    
    