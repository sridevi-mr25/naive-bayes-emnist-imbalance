

import numpy as np
from Task2_fit import CategoricalNaiveBayes

def predict_with_model(model, X):
    
    X = X.astype(float)
    N, D = X.shape
    C = len(model.classes_)

    #Getting log of class priors
    log_priors = np.log(model.class_priors_)

    
    log_probs = np.zeros((N, C))

    #Computing log-probability of each class for each sample
    for i, c in enumerate(model.classes_):
        theta_c = model.pixel_probs_[i]  
        # Calculate how likely each image is for this class by combining pixel-on(1) and pixel-off probabilities(0)
        log_likelihood = X @ np.log(theta_c) + (1 - X) @ np.log(1 - theta_c)
        log_probs[:, i] = log_priors[i] + log_likelihood

    #Pick the class with highest probability
    y_pred = model.classes_[np.argmax(log_probs, axis=1)]

    return y_pred
