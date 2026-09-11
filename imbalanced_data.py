import numpy as np

def sample_class_ratios(num_classes, alpha_class, rng=None):
    
    rng = np.random.default_rng(rng)
     #Generate class ratios using a Dirichlet distribution
    return rng.dirichlet(alpha_class * np.ones(num_classes))

def build_imbalanced_subset_indices(y, total_size, ratios, rng=None):
    rng = np.random.default_rng(rng)
    classes = np.unique(y)
    C = len(classes)
    # Calculate number of samples to pick from each class
    counts = np.floor(ratios * total_size).astype(int)
    while counts.sum() < total_size:
        counts[np.argmax(ratios - counts / total_size)] += 1
    while counts.sum() > total_size:
        counts[np.argmax(counts / total_size - ratios)] -= 1
    idxs = []
    # For each class, randomly picking required number of sample indices
    for c, k in zip(classes, counts):
        pool = np.where(y == c)[0]
        k = min(k, len(pool))  
        chosen = rng.choice(pool, size=k, replace=False)
        idxs.append(chosen)
        # Combine all chosen indices across classes
    return np.concatenate(idxs)
