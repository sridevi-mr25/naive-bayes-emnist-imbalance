import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from emnist import X_train, X_test, y_train, y_test
from Task2_fit import CategoricalNaiveBayes
from Task4_score import score

# Prepare data
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = np.array(y_train)
# Training sample sizes to test
train_sizes = np.arange(2000, 12000, 2000)   
# Fix β and vary α  
betas_fixed = 1                                
alphas = [1, 10, 100, 1000]   
# α_class values to simulate different imbalance levels                 
alpha_classes = [0.1, 0.2, 0.5, 1, 10, 100]    

colors = ["blue", "purple", "orange", "green"]  
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

# Loop through each imbalance levels of data
for j, alpha_class in enumerate(alpha_classes):
    ax = axes[j]
    # For each imbalance, test MAP models with different α values
    for i, alpha in enumerate(alphas):
        color = colors[i % len(colors)]
        train_means, val_means, train_stds, val_stds = [], [], [], []

        for size in train_sizes: 
            rng = np.random.default_rng(0)
            num_classes = len(np.unique(y_train))
            ratios = rng.dirichlet(alpha_class * np.ones(num_classes))
            counts = rng.multinomial(size, ratios)
  
            idxs = []
            for c, k in zip(np.unique(y_train), counts):
                pool = np.where(y_train == c)[0]
                k = min(k, len(pool))
                idxs.append(rng.choice(pool, size=k, replace=False))
            idxs = np.concatenate(idxs)
            X_subset, y_subset = X_train[idxs], y_train[idxs]

            
            fold_train, fold_val = [], []
            for tr_idx, va_idx in cv.split(X_subset, y_subset):
                X_tr, y_tr = X_subset[tr_idx], y_subset[tr_idx]
                X_val, y_val = X_subset[va_idx], y_subset[va_idx]

               
                map_model = CategoricalNaiveBayes(mode="MAP", alpha=alpha, beta=betas_fixed)
                map_model.fit(X_tr, y_tr)
                fold_train.append(score(map_model, X_tr, y_tr))
                fold_val.append(score(map_model, X_val, y_val))

            train_means.append(np.mean(fold_train))
            val_means.append(np.mean(fold_val))
            train_stds.append(np.std(fold_train))
            val_stds.append(np.std(fold_val))

        # Plot MAP curves
        ax.plot(train_sizes, val_means, "-", color=color, label=f"MAP Val α={alpha}")
        ax.plot(train_sizes, train_means, "--", color=color, label=f"MAP Train α={alpha}")
        ax.fill_between(train_sizes,
                        np.array(val_means) - np.array(val_stds),
                        np.array(val_means) + np.array(val_stds),
                        color=color, alpha=0.05)

# Plot MLE baseline curve   
    baseline_train, baseline_val = [], []
    for size in train_sizes:
        rng = np.random.default_rng(0)
        num_classes = len(np.unique(y_train))
        ratios = rng.dirichlet(alpha_class * np.ones(num_classes))
        counts = rng.multinomial(size, ratios)

        idxs = []
        for c, k in zip(np.unique(y_train), counts):
            pool = np.where(y_train == c)[0]
            k = min(k, len(pool))
            idxs.append(rng.choice(pool, size=k, replace=False))
        idxs = np.concatenate(idxs)
        X_subset, y_subset = X_train[idxs], y_train[idxs]

        fold_train, fold_val = [], []
        for tr_idx, va_idx in cv.split(X_subset, y_subset):
            X_tr, y_tr = X_subset[tr_idx], y_subset[tr_idx]
            X_val, y_val = X_subset[va_idx], y_subset[va_idx]
            mle = CategoricalNaiveBayes(mode="MLE")
            mle.fit(X_tr, y_tr)
            fold_train.append(score(mle, X_tr, y_tr))
            fold_val.append(score(mle, X_val, y_val))
        baseline_train.append(np.mean(fold_train))
        baseline_val.append(np.mean(fold_val))

    ax.plot(train_sizes, baseline_train, "k--", linewidth=2, label="MLE Train")
    ax.plot(train_sizes, baseline_val, "k-", linewidth=2, label="MLE Val")

    
    ax.set_title(f"α_class={alpha_class} (β=1)")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Avg Log-Likelihood")
    ax.grid(True)
    ax.legend(fontsize=6, loc="best")


plt.suptitle("Task 4.2: Varying Model α with Imbalanced Data (β=1)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
