import numpy as np
from Task2_fit import CategoricalNaiveBayes

def score(model, X, y):
   
    N, D = X.shape
    log_priors = np.log(model.class_priors_)
    total_log_likelihood = 0.0
    valid_count = 0
    eps = 1e-9  
    for n in range(N):
        c = y[n]  
        if c not in model.classes_:
            continue   
        c_idx = np.where(model.classes_ == c)[0][0]
        theta_c = model.pixel_probs_[c_idx]  
        log_p_x_given_y = (
            X[n] @ np.log(theta_c + eps) + (1 - X[n]) @ np.log(1 - theta_c + eps)
        )
        total_log_likelihood += log_priors[c_idx] + log_p_x_given_y
        valid_count += 1

    return total_log_likelihood / valid_count if valid_count > 0 else -np.inf
