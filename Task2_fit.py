
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CategoricalNaiveBayes(BaseEstimator, ClassifierMixin):
   
    def __init__(self, mode='MLE', alpha=1.0, beta=1.0):
       # mode: choose between MLE or MAP learning
        # alpha: class prior strength (Dirichlet)
        # beta: pixel prior strength (Beta) 
        self.mode = mode
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
       
        X = X.astype(float)
        y = np.array(y)
        N, D = X.shape 
        # Find unique classes in labels
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        # Count how many samples belong to each class
        class_counts = np.array([(y == c).sum() for c in self.classes_])

         # Computing class priors
        if self.mode.upper() == 'MLE':     
            # For MLE using the raw frequencies without prior  
            self.class_priors_ = class_counts / N

        elif self.mode.upper() == 'MAP':    
            self.class_priors_ = (class_counts + self.alpha ) / (
                N + C * (self.alpha)
            )
        else:
            raise ValueError("mode must be either 'MLE' or 'MAP'")

        # Computing conditional pixel probabilities
        self.pixel_probs_ = np.zeros((C, D))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]         
            n_c = X_c.shape[0]      
            pixel_sum = X_c.sum(axis=0)  

            if self.mode.upper() == 'MLE':
                # MLE estimate
                self.pixel_probs_[i] = pixel_sum / n_c
            else:
                #MAP estimate by applying prior Beta
                self.pixel_probs_[i] = (pixel_sum + self.beta ) / (
                    n_c + 2 * (self.beta)
                )

        #clipping values to avoid log(0) issues
        self.pixel_probs_ = np.clip(self.pixel_probs_, 1e-9, 1 - 1e-9)

        
        return self
