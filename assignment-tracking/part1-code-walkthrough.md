# Part 1 — Code Walkthrough & Concept Notes

This document explains every piece of code written for Part 1 of the assignment: what it does, why it was written that way, and the underlying concept. It is written for your own understanding — not for submission.

---

## 1.1 Data Loading and Exploration

### Loading MNIST
```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
```
`fetch_openml` pulls the MNIST dataset from the OpenML repository. The key parameters:
- `version=1` — pins to the canonical version so results are reproducible
- `return_X_y=True` — returns two separate arrays instead of a Bunch object (cleaner for ML workflows)
- `as_frame=False` — returns plain NumPy arrays instead of a Pandas DataFrame (required since our softmax implementation uses NumPy directly)

**What MNIST is:** 70,000 greyscale images of handwritten digits (0–9). Each image is 28×28 pixels, flattened into a row of 784 features. Every feature is a pixel intensity — an integer between 0 (black) and 255 (white). Labels are the digit the image represents, stored as strings `'0'`–`'9'`.

### Printing dataset statistics
```python
unique, counts = np.unique(y, return_counts=True)
```
`np.unique(y, return_counts=True)` returns the sorted unique classes and how many times each appears. This tells us whether the dataset is class-balanced — important because an imbalanced dataset would give misleading accuracy scores (a model that always predicts the majority class would still look accurate).

MNIST is roughly balanced: each digit has ~7,000 examples (9–11%). No imbalance problem here.

### Displaying sample images
```python
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
```
`X[i]` is a flat vector of 784 values. `.reshape(28, 28)` turns it back into the 2D image grid so `imshow` can render it. `cmap='gray'` uses a greyscale colour map. `axes.flatten()` turns the 2×5 grid of axes objects into a flat list so we can iterate it with a single loop. `ax.axis('off')` removes tick marks that would clutter small thumbnails.

---

## 1.2 Data Splitting

### Why 70 / 15 / 15?
The assignment requires three splits:
- **Training set (70%)** — the data the model sees and learns from
- **Validation set (15%)** — used *during* training to monitor overfitting and trigger early stopping; never used to update weights
- **Test set (15%)** — held out until the very end; gives an honest estimate of performance on truly unseen data

If you only had train/test, you'd have to guess hyperparameters blindly or risk "leaking" test information into your tuning decisions.

### Two-step split
```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```
`train_test_split` doesn't support three-way splits directly, so we do it in two steps:
1. Split 70% train / 30% temp
2. Split the 30% temp evenly → 15% val + 15% test

`random_state=42` makes the shuffle reproducible — running the notebook again produces identical splits.

### Why stratify?
`stratify=y` ensures each split has the same class proportions as the full dataset. Without it, random chance could give the test set more 1s and fewer 5s, which would distort per-class accuracy metrics.

For MNIST the classes are already roughly balanced (~10% each), so stratification has a small but measurable effect. The bar chart of class distributions verifies this visually — the three plots should look nearly identical.

### Class distribution bar chart
```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
splits = [('Train', y_train), ('Validation', y_val), ('Test', y_test)]
for ax, (name, labels) in zip(axes, splits):
    classes, counts = np.unique(labels, return_counts=True)
    ax.bar(classes, counts)
```
`zip(axes, splits)` pairs each subplot axis with one dataset split, iterating over all three simultaneously. `np.unique` recomputes per-split class counts. Setting `ax.set_xticks(classes)` ensures every digit label appears on the x-axis even if the string labels don't sort numerically by default.

---

## 1.3 Softmax Regression Implementation

### Feature normalisation
```python
X_train_norm = X_train / 255.0
```
Raw MNIST pixels are integers in [0, 255]. Without scaling, the dot product `X @ W` produces values in the hundreds, which `np.exp` turns into astronomically large numbers — floating-point overflow, resulting in `nan` gradients and a model that never learns.

Dividing by 255 maps every pixel to [0, 1]. The weight matrix then learns coefficients of a sensible magnitude, keeping gradients stable throughout training.

**Why not `StandardScaler`?** The lab solutions (`labsheet-02-sol`) use `StandardScaler` for general datasets. For pixel data, dividing by 255 is the idiomatic choice — it preserves the natural zero point (black = 0) and avoids computing per-pixel mean/variance statistics. `StandardScaler` would also work but is less conventional for image data.

---

### `SoftmaxRegression` class — method by method

The class follows the same sklearn-compatible conventions used in the lab solutions: `fit` returns `self`, exposes `predict` and `predict_proba`, and uses `np.random.default_rng` for reproducibility.

#### `__init__`
| Parameter | Value | Reason |
|---|---|---|
| `lr=0.1` | Learning rate | Empirically stable with [0,1]-normalised MNIST features |
| `max_epochs=200` | Max training epochs | Hard ceiling; early stopping fires well before this |
| `batch_size=256` | Mini-batch size | 192 batches per epoch over 49k samples — balances gradient noise vs. speed |
| `patience=15` | Early stopping patience | Allows brief plateaus without false-triggering |
| `random_state=42` | RNG seed | Reproducible mini-batch shuffling via `np.random.default_rng` |

---

#### `_add_bias`
```python
return np.c_[np.ones(X.shape[0]), X]
```
Prepends a column of ones, growing `X` from shape `(m, 784)` to `(m, 785)`. This lets a single weight matrix `W` of shape `(785, K)` absorb the bias term in its first row — no need for a separate `b` vector. The gradient formula then updates both weights and bias simultaneously in one matrix operation.

---

#### `_softmax`
```python
Z_shift = Z - Z.max(axis=1, keepdims=True)
exp_Z   = np.exp(Z_shift)
return exp_Z / exp_Z.sum(axis=1, keepdims=True)
```
**The concept:** softmax converts a vector of raw scores (logits) for K classes into a probability distribution that sums to 1. For class k: `p_k = exp(z_k) / Σ exp(z_j)`.

**The problem:** if any `z_k` is large (e.g. 500), `exp(500)` overflows to `inf`. Division of `inf/inf` gives `nan`.

**The fix:** subtract the row maximum before exponentiating. `exp(z_k - c) / Σ exp(z_j - c)` is mathematically identical (the constant `c` cancels), but now the largest value in each row is always `exp(0) = 1`, guaranteeing no overflow.

`keepdims=True` preserves the `(m, 1)` shape so the subtraction broadcasts correctly across all K columns.

---

#### `_one_hot`
```python
indices = np.array([self.class_to_idx_[label] for label in y])
Y = np.zeros((len(y), K))
Y[np.arange(len(y)), indices] = 1
```
Converts string labels like `['3', '7', '0', ...]` into a `(m, 10)` binary matrix. Row `i` has a 1 in the column for the true class, zeros everywhere else.

**Why this format?** The cross-entropy formula and gradient both use `Y_oh` to select the correct-class contribution. It is cleaner and faster than branching on individual labels inside the training loop.

The numpy fancy indexing `Y[np.arange(m), indices] = 1` sets exactly one 1 per row in a single vectorised call — much faster than a Python for loop over all 49,000 samples, and consistent with the vectorised style used throughout the labs.

---

#### `_cross_entropy`
```python
P = self._softmax(X_b @ self.W_)
return -np.mean(np.sum(Y_oh * np.log(P + 1e-15), axis=1))
```
Direct implementation of the assignment formula:
```
J(θ) = -1/m  Σ_i  Σ_k  y_k^(i)  log( p_k^(i) )
```
`Y_oh * np.log(P)` multiplies element-wise — because `Y_oh` is one-hot, only the term for the true class survives (all others multiply by 0). `np.sum(..., axis=1)` sums across classes per sample, `np.mean` averages across all m samples.

The `+ 1e-15` epsilon prevents `log(0) = -inf`, which can happen early in training when zero-initialised weights assign equal probability to all classes.

---

#### `fit` — the training loop

**Weight initialisation:** `self.W_ = np.zeros((n, K))` — all weights start at zero. This is safe for softmax: all classes begin with equal scores (uniform 10% probability). The first gradient update immediately breaks the symmetry because different training samples have different true labels.

**Mini-batch gradient descent:**
```
for each epoch:
    shuffle training data         ← new random order each epoch
    for each 256-sample batch:
        P    = softmax(Xb @ W)    ← forward pass: compute probabilities
        grad = Xb.T @ (P - Yb) / batch_size    ← gradient (from assignment brief)
        W   -= lr * grad          ← weight update
```
The gradient `(1/m) X^T (P - Y)` is the vectorised form of the per-class gradient from the assignment brief. It updates all 10 class weight vectors simultaneously — one matrix operation instead of K separate loops.

**Why shuffle each epoch?** Without shuffling, the model sees batches in the same fixed order every epoch. This can reinforce gradient oscillation patterns. Random shuffling ensures each batch is a fresh representative sample of the training data.

**Early stopping:**
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_W        = self.W_.copy()   # snapshot best weights
    no_improve    = 0
else:
    no_improve += 1
    if no_improve >= self.patience:
        break

self.W_ = best_W   # restore best weights after loop exits
```
Every epoch, we check whether the validation loss improved. If it hasn't improved for 15 consecutive epochs, we stop. After stopping, we restore `best_W` — the weights from the epoch with the *lowest* validation loss, not the weights from when training stopped. This is important: the last few epochs before stopping are already slightly overfitting, so the best-validation-epoch weights give a cleaner model.

---

#### `predict_proba` / `predict`
```python
def predict_proba(self, X):
    return self._softmax(self._add_bias(X) @ self.W_)

def predict(self, X):
    return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
```
`predict_proba` returns the `(m, 10)` probability matrix.  
`predict` takes the argmax across classes for each sample, then maps integer indices back to the original string labels (`'0'`–`'9'`) using `self.classes_`.

---

### Training results (from notebook output)
- Early stopping at **epoch 146** (best val loss: 0.2933)
- Train accuracy: **93.3%** | Validation accuracy: **92.0%**

These results are expected for a linear model on MNIST. Softmax regression draws linear decision boundaries in the 784-dimensional pixel space. Digits that are visually similar (4 vs. 9, 3 vs. 8) require curved boundaries to separate reliably, which a linear model cannot produce. The confusion matrix in Part 1.4 will confirm this pattern.

---

---

## 1.4 Scikit-Learn Logistic Regression Comparison

### Why compare to sklearn?
Our custom implementation is validated by comparing it to an industrial-grade solver. If both models — trained on the same normalised data — give similar accuracy on the test set, it confirms our gradient and softmax math is correct.

### Training the sklearn model
```python
from sklearn.linear_model import LogisticRegression
sklearn_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42)
sklearn_model.fit(X_train_norm, y_train)
```
Key parameters:
- `penalty=None` — disables L2 regularisation as required by the assignment. By default sklearn uses `penalty='l2'` which would shrink weights towards zero.
- `solver='lbfgs'` — Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm. This is a quasi-Newton second-order optimiser: it approximates the Hessian (curvature information) to take much larger, better-directed steps than our first-order gradient descent. It typically converges in tens of iterations vs. our hundreds of epochs.
- `max_iter=1000` — ensures convergence for the no-regularisation case (without a weight penalty, L-BFGS sometimes needs more iterations to fully converge on MNIST).
- `random_state=42` — reproducible result for the solver's internal shuffling.

**Why we use the same normalised data:** sklearn's L-BFGS is less sensitive to feature scale than our gradient descent, but using the same `X_train_norm` keeps the comparison fair — any accuracy difference is due to the optimiser, not the input scaling.

### Accuracy comparison table
```python
for name, clf in [('Custom Softmax Regression', model), ('sklearn LogisticRegression', sklearn_model)]:
    tr = accuracy_score(y_train, clf.predict(X_train_norm))
    te = accuracy_score(y_test,  clf.predict(X_test_norm))
    print(f"{name:<30}  {tr:>10.4f}  {te:>10.4f}")
```
Prints one row per model with train and test accuracy side by side. The f-string format `{tr:>10.4f}` right-aligns to 10 characters with 4 decimal places, producing a readable table without any library. `accuracy_score(y_true, y_pred)` is `correct_predictions / total_predictions`.

### Confusion matrices
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test_norm), ax=axes[0], colorbar=False)
ConfusionMatrixDisplay.from_predictions(y_test, sklearn_model.predict(X_test_norm), ax=axes[1], colorbar=False)
```
`ConfusionMatrixDisplay.from_predictions` takes true labels and predicted labels, builds the 10×10 matrix, and renders it as a heatmap with class labels on both axes. Displaying both models side by side on a single figure makes misclassification pattern comparison immediate — you can see at a glance whether the two models struggle with the same digit pairs.

**What to look for in a MNIST confusion matrix:**
- The diagonal = correct predictions (should be dark/high values)
- Off-diagonal = misclassifications; bright cells = common errors
- Expected hard pairs for linear models: **4↔9** (similar vertical strokes), **3↔8** (curved top), **5↔6**, **7↔1**

`colorbar=False` removes the colour scale sidebar to save space when displaying two matrices side by side.

### Classification report
```python
print(classification_report(y_test, model.predict(X_test_norm), digits=4))
```
`classification_report` prints a table with four columns per class:
- **Precision** — of all samples predicted as class k, what fraction were actually class k
- **Recall** — of all samples that are class k, what fraction did the model find
- **F1-score** — harmonic mean of precision and recall; penalises imbalance between the two
- **Support** — number of test samples with that true label

`digits=4` gives 4 decimal places for finer comparison between models.

**Macro avg vs weighted avg:**
- Macro avg: simple mean across classes — treats every class equally regardless of frequency
- Weighted avg: mean weighted by support (number of samples per class) — closer to overall accuracy

For MNIST (roughly balanced classes) these two averages should be nearly identical.

### Expected results
Both models are fundamentally **linear classifiers** — they learn a hyperplane in the 784-dimensional pixel space. The accuracy gap between them reflects optimiser quality, not model capacity:
- sklearn's L-BFGS will likely be ~1–2% higher on the test set
- Both models will make similar errors on visually similar digit pairs
- The confusion matrices should look qualitatively alike — the main difference will be in the magnitude of off-diagonal cells, not their location

---

### Lab style alignment (code review)

Checked against all three lab solutions (`labsheet-02-sol`, `labsheet-03-sol`, `simple-learning-model-sol`):

| Convention | Lab solutions | Our implementation |
|---|---|---|
| Class-based model | ✓ `Simple_Model`, `LogisticRegressionSGD` | ✓ `SoftmaxRegression` |
| `fit()` returns `self` | ✓ | ✓ |
| `predict_proba()` + `predict()` | ✓ | ✓ |
| `np.random.default_rng` for RNG | ✓ | ✓ |
| `random_state=42` | ✓ | ✓ |
| Vectorised NumPy (no per-sample loops in hot path) | ✓ | ✓ |
| Sklearn-compatible API | ✓ | ✓ |
| Concise code, explanation in markdown cells | ✓ | ✓ |

**Change made during review:** the original `_one_hot` used a Python `for` loop over all 49,000 samples (setting one element at a time). Replaced with numpy fancy indexing `Y[np.arange(m), indices] = 1` — same result, fully vectorised, consistent with lab style.

---

## 2.1 Dataset Generation

### The formula
The assignment specifies:
$$y = \sum_{k=0}^{n} a_k x^k + \varepsilon$$
- $x$ sampled uniformly from $[-3, 3]$
- Each coefficient $a_k \sim \mathcal{U}(0, 1)$ — all positive, so the polynomial always opens upward
- $\varepsilon \sim \mathcal{N}(0, 1)$ — unit Gaussian noise added independently to each point

### `generate_polynomial_dataset`
```python
def generate_polynomial_dataset(degree, n_samples, random_state=42):
    rng = np.random.default_rng(random_state)
    x   = rng.uniform(-3, 3, n_samples)
    a   = rng.uniform(0, 1, degree + 1)
    y   = sum(a[k] * x**k for k in range(degree + 1)) + rng.standard_normal(n_samples)
    return x.reshape(-1, 1), y
```

**`rng.uniform(-3, 3, n_samples)`** — draws `n_samples` x values independently from a continuous uniform distribution over [-3, 3]. This is the input feature — 1D, so `x` is reshaped to `(-1, 1)` (a column vector) so sklearn's SVR accepts it without modification.

**`rng.uniform(0, 1, degree + 1)`** — draws `degree + 1` coefficients, one per power. `a[0]` is the constant term, `a[1]` is the linear coefficient, and so on up to `a[degree]`. All are drawn from [0, 1] so the polynomial shape is always positive.

**`sum(a[k] * x**k for k in range(degree + 1))`** — directly implements the assignment formula as a Python generator expression. `x**k` raises every element of the numpy array `x` to the power `k` (element-wise). The `sum(...)` accumulates across k, which numpy handles as vectorised addition across arrays. For degree=3 this computes `a[0] + a[1]*x + a[2]*x² + a[3]*x³` for all 1000 values of `x` at once.

**`rng.standard_normal(n_samples)`** — draws noise from $\mathcal{N}(0, 1)$. Adding this after the polynomial computation ensures the noise is independent of x.

**`x.reshape(-1, 1)`** — sklearn's `SVR.fit(X, y)` expects `X` to be 2D (shape `(n_samples, n_features)`). For our 1D problem that's `(1000, 1)`. The `-1` in reshape means "infer this dimension automatically".

**`random_state` parameter** — passed to `np.random.default_rng` so the same call always produces the same dataset. The function defaults to 42 but the three experiment datasets use seeds 10, 20, 30 so they're genuinely independent from each other.

### Why different seeds for the three datasets?
```python
X_lin,  y_lin  = generate_polynomial_dataset(degree=1, n_samples=1000, random_state=10)
X_quad, y_quad = generate_polynomial_dataset(degree=2, n_samples=1000, random_state=20)
X_cub,  y_cub  = generate_polynomial_dataset(degree=3, n_samples=1000, random_state=30)
```
Using seeds 10, 20, 30 ensures each dataset has:
- Different x sample positions
- Different polynomial coefficients
- Different noise realisations

If we used the same seed for all three, they'd share the same x values and same noise, meaning every observed difference between datasets would be purely due to the polynomial shape — a controlled but artificial setup. Independent seeds give a more realistic simulation of three distinct experiments.

### Train/test split
```python
X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
```
Standard 80/20 split. The assignment doesn't specify a ratio — 80/20 is the default convention for regression tasks. `test_size=0.2` on 1000 samples gives 800 training and 200 test points per dataset.

Note that unlike the Part 1 MNIST split, **no validation set is needed here** — the hyperparameter search in 2.3 uses cross-validation on the training set, so a separate validation split is not required.

### Verification plots
```python
configs = [('Linear (degree=1)', 1, 10), ('Quadratic (degree=2)', 2, 20), ('Cubic (degree=3)', 3, 30)]
for ax, (title, degree, seed) in zip(axes, configs):
    X_demo, y_demo = generate_polynomial_dataset(degree, n_samples=300, random_state=seed)
    ax.scatter(X_demo, y_demo, s=8, alpha=0.5)
```
300 samples are used for the demo (rather than 1000) because scatter points overlap less and the shape is still clearly visible. `s=8` gives small dots, `alpha=0.5` adds transparency so dense regions appear darker. The plots should show:
- **degree=1**: a noisy linear band with positive slope (all $a_k > 0$)
- **degree=2**: a noisy upward-opening parabola
- **degree=3**: a noisy S-curve, more complex — the cubic term dominates at the edges

---

## Learning Module Tab — Build Prompt

The following prompt is designed to be given to a separate Claude instance that has access to your ML learning module codebase. Provide it with this walkthrough file and the assignment notebook (`23334375_assignment_1.ipynb`) as context.

---

```
I am building an interactive learning tab in my custom ML learning module. I will give you two files for context:

1. `C:\Users\james\Documents\university\cits5508\CITS5508-assignment-1\23334375_assignment_1.ipynb` — a Jupyter notebook for a UWA CITS5508 (Machine Learning) university assignment. It currently has Part 1 implemented: MNIST data loading/exploration, stratified train/val/test splitting, a from-scratch softmax regression classifier (NumPy only, mini-batch gradient descent, early stopping), and a comparison against sklearn's LogisticRegression.

2. `C:\Users\james\Documents\university\cits5508\CITS5508-assignment-1\assignment-tracking\part1-code-walkthrough.md` — a detailed technical walkthrough of every section of code written so far: what it does, why each design decision was made, and the underlying ML concepts.

Before building anything, read the existing module structure thoroughly — understand how existing tabs/pages/components are built, what frameworks/libraries are used, how content is wired up, and what patterns are followed. Match those patterns exactly for the new tab.

Once you understand the structure, build a new interactive tab called something like "CITS5508 Assignment 1" or "Softmax & SVR" (use whatever naming convention the module already uses). The tab should cover the following content, structured as progressive sections the user can step through:

---

SECTION 1 — MNIST & The Classification Problem
- What MNIST is: 70k 28×28 greyscale digit images, features = pixel intensities [0–255], labels = digit class '0'–'9'
- Why we flatten images to vectors (784 features per sample)
- Interactive: show a grid of sample MNIST images with their labels (use the data from the notebook if possible, otherwise use static examples)
- Concept check: what does the class distribution look like? Why does class balance matter?

SECTION 2 — Train / Validation / Test Splits
- Why three splits are needed (training, monitoring overfitting, final evaluation)
- What stratification means and why it matters for classification
- Interactive: a visual showing how 70,000 samples are divided, with class distribution bars for each split
- Concept check: what would happen if you tuned hyperparameters against the test set?

SECTION 3 — Softmax Regression from Scratch
- The softmax function: converting raw scores (logits) to probabilities. Show the formula. Show WHY we subtract the row max (numerical stability — overflow prevention).
- Cross-entropy loss: show the formula, explain that only the true class term survives because Y is one-hot
- The gradient: show ∂J/∂θ^(k) = (1/m) X^T (P - Y), explain this is the vectorised form
- Mini-batch gradient descent: explain the epoch/batch loop, why we shuffle each epoch
- Early stopping: explain patience, best-weight restoration
- Interactive: an annotated walkthrough of the `SoftmaxRegression` class methods (_add_bias, _softmax, _one_hot, _cross_entropy, fit, predict) — one method at a time, with explanation alongside the code
- Interactive: show the training/validation loss curve from the notebook (epoch 146 early stop, best val loss 0.2933, train acc 93.3%, val acc 92.0%)

SECTION 4 — sklearn LogisticRegression Comparison
- What L-BFGS is vs gradient descent (second-order vs first-order, curvature information, faster convergence)
- Why `penalty=None` is needed (default sklearn uses L2 regularisation)
- Why both models are still fundamentally LINEAR classifiers — the same decision boundary limitations apply
- Interactive: show the accuracy comparison table (custom vs sklearn, train vs test)
- Interactive: show/render the side-by-side confusion matrices. Highlight the hard digit pairs (4↔9, 3↔8)
- Interactive: the classification report table — let the user click a digit class to highlight its row in both models' reports
- Concept check: why do both models struggle with the same digit pairs even though sklearn's optimiser is better?

SECTION 5 — What's Linear vs Non-Linear
- Explain the fundamental limitation of linear classifiers on MNIST: a hyperplane in 784D pixel space cannot separate all digit classes perfectly
- Which digit pairs need a curved boundary and why (visual similarity)
- Bridge to future content: this is why neural networks / kernel methods (like SVR with RBF kernel, covered in Part 2) exist

---

General requirements for the tab:
- Match the exact visual style, component structure, and navigation patterns of the existing tabs in the module
- All math formulas should render properly (use whatever formula rendering the module already uses)
- Code snippets should be syntax-highlighted (use whatever the module already uses)
- Each section should have a short "key takeaway" summary card at the bottom
- Progress should be tracked if the module has any progress/completion tracking
- Do not use placeholder content — all explanations, numbers, and code snippets should come directly from the two files I have provided

Ask me any questions about the module's architecture before you start building.
```
