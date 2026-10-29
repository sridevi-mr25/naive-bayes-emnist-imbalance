# Naive Bayes on EMNIST: When Do Priors Help?

A Naive Bayes classifier for handwritten letters and digits, built from scratch with NumPy. It is trained two ways: with plain frequency counts (MLE), and with priors that smooth those counts (MAP). The question is when the priors pay off, as the amount of training data and the class balance change.

*Rutgers machine learning mini project, October 2025. Individual work.*

## The problem

EMNIST "balanced" has 47 classes of handwritten characters, digits plus upper- and lower-case letters, as 28 × 28 images. Each pixel is binarized to on or off. The model treats the 784 pixels as independent given the class:

- **MLE** estimates each class's frequency and each pixel's "on" rate directly from counts.
- **MAP** adds two priors. A Dirichlet prior **α** on class frequencies pulls them toward uniform. A Beta prior **β** on each pixel's "on" rate stops it from reaching exactly 0 or 1.

To test robustness to class imbalance, the experiments also draw training sets whose class mix comes from a Dirichlet distribution. A small **α_class** (for example 0.1) gives a few dominant classes and many rare ones. A large α_class (100) gives a near-even mix.

## Experiments

All comparisons use average log-likelihood with 5-fold shuffle splits. Higher is better.

| Experiment | What varies | Script | Result plot |
|---|---|---|---|
| 3.1 | Class prior α = 1, 10, 50, 100, 200 (β = 1) | `Beta_task3.py` | `Task3.1 Result.png` |
| 3.2 | Pixel prior β = 1, 2, 10, 100 (α = 1) | `Alpha_task3.py` | `Task3.2 Result.png` |
| 4.1 | β under six imbalance levels | `task4_beta.py` | `Task4.1 Result.png` |
| 4.2 | α under six imbalance levels | `task4_alpha.py` | `Task4.2 Result.png` |

> The two Task 3 script names are swapped: `Alpha_task3.py` varies β, and `Beta_task3.py` varies α.

## Findings

**1. MLE overfits when data is scarce, and MAP fixes it.** With 1,000 training images, MLE scores about −212 on training data but only about −303 on validation. MAP with β = 1 scores about −256 on validation. The gap closes as data grows, and by about 9,000 images the two models are nearly level. The cause is simple: if a pixel is never "on" for a class in training, MLE treats it as impossible for that class. Each unseen pixel then costs about 21 points of log-likelihood when it does appear. The β prior removes those zeros.

**2. A light pixel prior works best.** β = 1 or 2 gives the best validation score at every training size. Heavier smoothing underfits: at 9,000 images, β = 10 scores about −250 and β = 100 about −373, against about −233 for β = 1.

**3. The class prior barely matters.** Every α from 1 to 1,000 produces almost the same curve, in both the balanced and imbalanced experiments. The class prior adds one term to each prediction, while the pixels add 784, so it is easily outweighed.

**4. MAP holds up better under imbalance.** The skewed class mixes widen MLE's validation gap. At the strongest imbalance (α_class = 0.1) with 10,000 images, MLE scores about −250 against about −237 for MAP. Once classes are close to balanced (α_class ≥ 10), the two converge.

**Recommendation:** use MAP with β ≈ 1–2. The choice of α makes little difference.

## Limitations

- **Accuracy isn't reported.** Every comparison uses average log-likelihood. `run_models.py` scores both models on the held-out test set but doesn't print the results or a classification accuracy, which is what most readers would ask for first.
- **There is a large unused test set.** The split keeps 90% of EMNIST for testing, and none of the plots use it.
- **The independence assumption is strong.** Naive Bayes treats neighbouring pixels as unrelated, which handwriting clearly isn't. That caps how well any setting of α and β can do.

## How to run

```bash
pip install numpy scikit-learn matplotlib torch torchvision

python visualisation_emnist.py   # sample grid of all 47 classes
python run_models.py             # fit MLE and MAP models
python Beta_task3.py             # Experiment 3.1 (varies α)
python Alpha_task3.py            # Experiment 3.2 (varies β)
python task4_beta.py             # Experiment 4.1
python task4_alpha.py            # Experiment 4.2
```

EMNIST downloads to `~/data` on first run, about 500 MB. The imbalance experiments retrain many models and can take several minutes.

## Repository contents

| File | Contents |
|---|---|
| `emnist.py` | Loads EMNIST balanced, binarizes pixels, splits 10% train / 90% test |
| `Task2_fit.py` | `CategoricalNaiveBayes`: MLE and MAP fitting, scikit-learn compatible |
| `Task2_predict.py` | Prediction with log-probabilities |
| `Task2_score.py`, `Task4_score.py` | Average log-likelihood; the Task 4 version handles classes missing from a sample |
| `imbalanced_data.py` | Draws imbalanced training subsets from Dirichlet class ratios |
| `ML mini project results/` | Result plots and the written discussion (Tasks 5.1 and 5.2) |
