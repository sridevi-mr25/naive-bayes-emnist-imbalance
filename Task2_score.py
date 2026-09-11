
import numpy as np
from Task2_fit import CategoricalNaiveBayes

def score(model, X, y):
   
    N, D = X.shape
    # Taking the log of class priors
    log_priors = np.log(model.class_priors_)
    total_log_likelihood = 0.0

    for n in range(N):
        c = y[n]                          
        theta_c = model.pixel_probs_[c]   

        # Calculate how likely the image is given its true class
        # Combines probabilities of each pixel being on (1) or off (0)
        log_p_x_given_y = X[n] @ np.log(theta_c) + (1 - X[n]) @ np.log(1 - theta_c)

        
        total_log_likelihood += log_priors[c] + log_p_x_given_y

    # Return the average log-likelihood across all samples
    return total_log_likelihood / N
