import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from emnist import X_train, X_test, y_train, y_test
from Task2_fit import CategoricalNaiveBayes
from Task2_score import score

# Prepare data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test  = X_test.reshape(X_test.shape[0], -1)

# Defining the number of training samples for each iteration
train_sizes = np.arange(1000, 10000, 1000)  
#Test different pixel prior (α) values while keeping β fixed
alphas = [1, 10, 50, 100, 200]
beta_fixed = 1

colors = ["purple", "blue", "teal", "green", "gold"]

# Create 5 random train/validation splits
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

plt.figure(figsize=(10, 6))

# --- store MLE baseline once ---
baseline_train_MLE, baseline_val_MLE = None, None

for i, alpha in enumerate(alphas):
    color = colors[i]
    train_means, val_means, train_stds, val_stds = [], [], [], []
    train_means_MLE, val_means_MLE, train_stds_MLE, val_stds_MLE = [], [], [], []

    for size in train_sizes:
        # Take a subset of the training data
        X_subset, y_subset = X_train[:size], y_train[:size]

        fold_train_MAP, fold_val_MAP = [], []
        fold_train_MLE, fold_val_MLE = [], []
         # Perform cross-validation
        for train_idx, val_idx in cv.split(X_subset, y_subset):
            X_tr, y_tr = X_subset[train_idx], y_subset[train_idx]
            X_val, y_val = X_subset[val_idx], y_subset[val_idx]

            # Train MAP model with priors
            map_model = CategoricalNaiveBayes(mode="MAP", alpha=alpha, beta=beta_fixed)
            map_model.fit(X_tr, y_tr)
            fold_train_MAP.append(score(map_model, X_tr, y_tr))
            fold_val_MAP.append(score(map_model, X_val, y_val))

            # Train MLE model
            mle_model = CategoricalNaiveBayes(mode="MLE")
            mle_model.fit(X_tr, y_tr)
            fold_train_MLE.append(score(mle_model, X_tr, y_tr))
            fold_val_MLE.append(score(mle_model, X_val, y_val))

        # Recording average and standard deviation of log-likelihoods across folds
        train_means.append(np.mean(fold_train_MAP))
        val_means.append(np.mean(fold_val_MAP))
        train_stds.append(np.std(fold_train_MAP))
        val_stds.append(np.std(fold_val_MAP))

        train_means_MLE.append(np.mean(fold_train_MLE))
        val_means_MLE.append(np.mean(fold_val_MLE))
        train_stds_MLE.append(np.std(fold_train_MLE))
        val_stds_MLE.append(np.std(fold_val_MLE))

    # Plot MAP learning curve
    train_means, val_means = np.array(train_means), np.array(val_means)
    train_stds,  val_stds  = np.array(train_stds),  np.array(val_stds)
    plt.plot(train_sizes, train_means, "--", color=color, label=f"α={alpha} train")
    plt.plot(train_sizes, val_means, "-", color=color, label=f"α={alpha} val")
    plt.fill_between(train_sizes, train_means - train_stds, train_means + train_stds, color=color, alpha=0.08)
    plt.fill_between(train_sizes, val_means - val_stds, val_means + val_stds, color=color, alpha=0.08)
    # Save MLE baseline only once
    if baseline_train_MLE is None:
        baseline_train_MLE = train_means_MLE
        baseline_val_MLE = val_means_MLE
# Plot MLE baseline
plt.plot(train_sizes, baseline_train_MLE, "--", color="black", linewidth=2, label="MLE train (baseline)")
plt.plot(train_sizes, baseline_val_MLE, "-", color="black", linewidth=2, label="MLE val (baseline)")

plt.xlabel("Training samples")
plt.ylabel("Average Log-Likelihood")
plt.title(" Learning Curves for different  α values (β=1.0) ")
plt.legend(fontsize=8, loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
