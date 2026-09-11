
import numpy as np
from emnist import X_train, X_test, y_train, y_test
from Task2_fit import CategoricalNaiveBayes
from Task2_predict import predict_with_model 
from Task2_score import score

# Prepare the data by Flattening 28x28 images into 1D arrays(each image has 784 pixels)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train both models
mle_model = CategoricalNaiveBayes(mode='MLE')
mle_model.fit(X_train, y_train)
map_model = CategoricalNaiveBayes(mode='MAP', alpha=1.0, beta=2.0)
map_model.fit(X_train, y_train)

# Predict labels for test data
y_pred_mle = predict_with_model(mle_model, X_test)
y_pred_map = predict_with_model(map_model, X_test)

# Evaluate model performance
mle_score = score(mle_model, X_test, y_test)
map_score = score(map_model, X_test, y_test)



