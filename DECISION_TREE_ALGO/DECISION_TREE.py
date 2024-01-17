from collections import Counter
import numpy as np

class node:
    def __init__(self, feature=None , threshold= None , left= None , right=None , * , value = None):
        self.feature = feature
        self.threshold= threshold
        self.left = left
        self.right = right 
        self.value = value
        
    def _is_a_leaf(self):
        return self.value is not None
        
        
class Decision_tree:
    def __init__(self, n_features=None, max_dept=100 ,min_self_split=2 ):
        self.n_features = n_features
        self.max_dept= max_dept
        self.min_self_split = min_self_split
        self.root = None
        

    def fit(self,X, y):
        self.n_features = X.shape[1] if not self.n_features  else min(X.shape[1],self.n_features)
        self.root = self.grow(X,y)
        
        
    def grow(self ,X ,y, dept = 0):
        n_samples ,n_feats = X.shape
        n_labels = len(np.unique(y))
        feat_idxs= np.random.choice (n_feats, self.n_features, replace = False)
        #If a leaf then give values
        if (n_samples<self.min_self_split or n_labels==1 or dept>=self.max_dept):
            leaf_value = self._most_common_label(y)
            return node(value=leaf_value)
        
        #find best features
        best_feature , best_threshold = self._best_split(X,y,feat_idxs)
        
        #give birth to child node
        left_idxs, right_idxs = self._split(X[:,best_feature],best_threshold)
        left = self.grow(X[left_idxs,:],y[left_idxs],dept+1)
        right = self.grow(X[right_idxs,:],y[right_idxs],dept+1)
        return node(best_feature,best_threshold,left,right)
        
        
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1 
        op_feat , op_threshold = None,None
        
        for feat_idx in feat_idxs:
            X_coloumn = X[:,feat_idx]
            thresholds = np.unique(X_coloumn)
            
            for thr in thresholds:
                gain = self._information_gain(y, X_coloumn, thr)
                if (gain>= best_gain):
                    best_gain = gain
                    op_feat = feat_idx
                    op_threshold = thr
                    
        return op_feat,op_threshold
    
    
    def _information_gain(self, y, x, thr):
        parents_entropy = self._entropy(y)
        
        left_indx, right_indx = self._split(x, thr)
        
        if len(left_indx)==0 or len(right_indx)==0:
            return 0
        
        n = len(y)
        n_l , n_r = len(left_indx), len(right_indx)
        e_l , e_r = self._entropy(y[left_indx]),self._entropy(y[right_indx])
        child_entropy = (n_l/n)* e_l + (n_r/n)* e_r
        
        return parents_entropy- child_entropy
    
    def _entropy(self, y):
        his = np.bincount(y)
        ps = his/ len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])
        
        
    def _split(self, X_column, thr):
        left_idxs = np.argwhere(X_column <= float(thr)).flatten()
        right_idxs = np.argwhere(X_column> float(thr)).flatten()
        return left_idxs,right_idxs
                    
            
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value   
        
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])


    def _traverse_tree(self, x, node):
        if node._is_a_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)