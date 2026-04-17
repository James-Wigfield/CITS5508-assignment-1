# Plain-English Explainer: Assignment 1

This document explains every section of the notebook in plain, simple terms. No assumed prior knowledge. If you can read a recipe, you can read this.

---

## The Big Picture

The assignment has two completely separate jobs:

1. **Part 1** — Teach a computer to read handwritten digits (0–9) using a technique called Softmax Regression. We build it from scratch first, then compare it to a ready-made library version.
2. **Part 2** — Teach a computer to fit a curve through noisy data using Support Vector Regression (SVR). We try different types of curves and tune the settings.

---

## Part 1: Softmax Regression on MNIST

### What is MNIST?

MNIST is a dataset of 70,000 small black-and-white images of handwritten digits. Each image is 28×28 pixels. Computers see an image as a grid of numbers — each pixel has a brightness value from 0 (black) to 255 (white). So each image becomes a list of 784 numbers (28 × 28 = 784).

The job: look at those 784 numbers and predict which digit (0 through 9) was written.

---

### 1.1 — Loading and Exploring the Data

**What the code does:**
- Downloads the MNIST dataset using scikit-learn's `fetch_openml`.
- Prints basic info: how many samples, how many features, what the labels look like.
- Shows the first 10 images as actual pictures.
- Shows one example of each digit (0–9) so you can see the variety.

**Plain English:**
Think of this like opening a box of 70,000 index cards. Each card has a tiny photo of a handwritten number on one side, and the answer (what digit it is) written on the other side. We're just peeking through the box to see what we're working with.

The pixel values are stored as floats (0.0–255.0). The labels are stored as *strings* — `'0'`, `'1'`, ..., `'9'` — not integers. This matters later.

---

### 1.2 — Splitting the Data

**What the code does:**
- Splits the 70,000 images into three groups:
  - **Training set** (70% = ~49,000 images): the model learns from these.
  - **Validation set** (15% = ~10,500 images): used to check progress *during* training and tune settings.
  - **Test set** (15% = ~10,500 images): locked away until the very end for a fair final score.
- Uses **stratification**, which means each split has roughly the same proportion of each digit.
- Plots bar charts showing the class distribution in each split.

**Plain English:**
Imagine you have a pile of 70,000 flashcards. You split them into three piles:
- A big "study pile" you use to learn from.
- A "practice test" pile you use to check how you're doing while studying.
- A "real exam" pile you don't look at until you're done studying.

**Why stratify?** If you split randomly without stratification, you might get lucky and have way more 1s than 5s in your training set (since 1 is the most common digit and 5 is the least common). That would give your model a skewed view of reality. Stratification guarantees each pile has roughly the same proportion of each digit — a fair sample.

---

### 1.3 — Building Softmax Regression from Scratch

This is the main event of Part 1. We build the classifier ourselves using only NumPy (no sklearn).

#### Step 0: Normalisation

```python
X_train_norm = X_train / 255.0
```

Pixel values go from 0–255 down to 0.0–1.0. This is called **normalisation**. Why? Because large numbers make gradient descent (the learning algorithm) behave badly — it takes huge, unstable steps. Keeping values small and in the same range makes learning smooth and fast.

---

#### The SoftmaxRegression Class — Piece by Piece

**`__init__` (the settings)**

When you create a `SoftmaxRegression` object, you configure it:
- `lr` = learning rate. How big a step the model takes when it's wrong. Too big → it overshoots. Too small → it takes forever to learn.
- `max_epochs` = maximum number of passes through the training data. An *epoch* is one complete pass.
- `batch_size` = how many images to look at before adjusting the weights. We don't update after every single image (too slow) or after all 49,000 (too inaccurate). We do it in batches of 256.
- `patience` = early stopping. If the model stops improving for this many epochs, we quit early.
- `print_every` = how often to print a progress update.

---

**`_add_bias` — Adding a bias column**

```python
return np.c_[np.ones(X.shape[0]), X]
```

We prepend a column of 1s to the data. This is a mathematical trick: instead of having a separate bias variable `b`, we absorb it into the weight matrix `W`. Every row now starts with `1`, so the first column of `W` acts as the bias. Fewer separate variables to track = cleaner code.

---

**`_softmax` — Converting raw scores to probabilities**

The model produces a raw score for each digit class. But we want *probabilities* (e.g., "60% chance it's a 7, 30% chance it's a 1, 10% other"). Softmax converts scores to probabilities:

```
probability of class k = exp(score_k) / sum of all exp(scores)
```

The `exp` (e^x) function makes differences between scores bigger, and dividing by the sum makes everything add up to 1 (proper probabilities).

**Why subtract the row max first?** If scores are very large (e.g., 1000), `exp(1000)` is astronomically huge and computers can't handle it (overflow error). Subtracting the max first keeps the numbers manageable without changing the final probabilities — it's just a safety trick.

---

**`_one_hot` — Encoding the labels**

The label `'3'` isn't useful for maths. We convert it to a vector: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. That's a "one-hot" vector — all zeros except a 1 at the position of the correct class. This lets us compare the model's probability predictions directly against the correct answer.

---

**`_cross_entropy` — Measuring how wrong the model is**

Cross-entropy loss is the "how wrong am I?" score. It works like this:
- The model predicts a probability for each class.
- We look at the probability it gave to the *correct* class.
- We take `-log(that probability)`.

If the model was 99% confident and correct → loss is tiny.
If the model was 1% confident and correct (basically wrong) → loss is huge.

We average this over all training examples to get one number.

---

**`fit` — The actual learning loop**

This is where learning happens. Here's the flow:

1. **Set up weights** (`W_`): a big matrix of zeros, shape `(785, 10)` — 784 features + 1 bias, for each of 10 digit classes. All zeros at the start = completely ignorant model.

2. **For each epoch:**
   - **Shuffle** the training data (so mini-batches are different each time — prevents the model from memorising the order).
   - **Loop through mini-batches** of 256 images:
     - Compute predictions using current weights (softmax).
     - Compute the gradient — a matrix that tells us which direction to nudge `W_` to reduce the loss. Formula: `(1/m) * X^T * (P - Y)`. This comes directly from calculus (chain rule on the cross-entropy loss).
     - Update weights: `W_ -= lr * gradient` (a small step in the direction that reduces loss).
   - Compute training loss and validation loss.
   - **Early stopping check**: if validation loss didn't improve, increment a counter. If it hasn't improved for `patience` epochs, stop and use the best weights found so far.

**Why validation loss and not training loss for early stopping?** Training loss almost always keeps going down. Validation loss tells you when the model starts *memorising* the training data instead of *learning* from it (overfitting). When validation loss stops improving, we've hit peak generalisation.

---

**`predict_proba` and `predict`**

`predict_proba` runs the forward pass (add bias → multiply by W → softmax) to get probabilities.

`predict` just takes the class with the highest probability (argmax) and returns its label.

---

#### Training the Model

```python
model = SoftmaxRegression(lr=0.1, max_epochs=200, batch_size=256, patience=15, random_state=42)
model.fit(X_train_norm, y_train, X_val_norm, y_val)
```

We create and train the model. During training it prints progress every 10 epochs showing the train and validation loss. It stops early if validation loss plateaus for 15 consecutive epochs.

---

#### Loss Curve Plot

After training, we plot training loss vs. validation loss over epochs. In a healthy model:
- Both curves go down over time (the model is learning).
- Validation loss is slightly higher than training loss (expected — the model hasn't seen the validation data).
- The curves eventually flatten out and converge — this is when learning has finished.

If validation loss starts *going back up* while training loss keeps going down, the model is overfitting (memorising rather than generalising). Early stopping prevents this.

---

### 1.4 — Comparing Against sklearn's Logistic Regression

**What the code does:**
- Trains sklearn's `LogisticRegression` on the same normalised data.
- `penalty=None` disables regularisation so it's a fair comparison with our custom model (which also has no regularisation).
- Prints accuracy for both models on train and test sets.
- Shows side-by-side confusion matrices.
- Prints full classification reports.

**What is a confusion matrix?**
A 10×10 grid. The rows are "what the correct label was", the columns are "what the model predicted". The diagonal is correct predictions. Off-diagonal entries are mistakes — and *which* digits get confused for each other is interesting (e.g., 4 and 9 look similar).

**What is a classification report?**
For each digit class, it shows:
- **Precision** — of all the times the model said "this is a 3", what fraction was actually a 3?
- **Recall** — of all the actual 3s, what fraction did the model correctly identify?
- **F1-score** — a single number combining precision and recall (harmonic mean).

**Why compare to sklearn?**
It validates our from-scratch implementation. If both models achieve similar accuracy, we can be confident our maths and code are correct.

---

## Part 2: Support Vector Machine Regression

### What is SVR?

Support Vector Regression (SVR) is a way to fit a curve through data. Unlike regular linear regression (which draws a straight line minimising total error), SVR:
- Creates a "tube" around the fitted curve.
- Points *inside* the tube are considered close enough — they contribute zero penalty.
- Points *outside* the tube are penalised.

This makes SVR more robust to outliers than regular regression.

---

### 2.1 — Generating Synthetic Polynomial Datasets

**What the code does:**
- Defines `generate_polynomial_dataset(degree, n_samples)`.
- Generates three datasets: linear (degree 1), quadratic (degree 2), cubic (degree 3).
- Plots them to visually verify the function works.
- Splits each into 80% train / 20% test.
- Prints summary statistics.

**Plain English:**
We're making up fake data that follows a polynomial curve, then adding random noise to it. Like drawing a curve on paper, then shaking the pen slightly while drawing — the points are near the curve but not exactly on it.

The formula: `y = a₀ + a₁x + a₂x² + ... + aₙxⁿ + noise`

- `x` values are random numbers between -3 and 3.
- The coefficients (`a₀, a₁, ...`) are random numbers between 0 and 1.
- The noise is drawn from a standard normal distribution (bell curve centred at 0).

Why make up data? Because we *know* the true shape, so we can test whether SVR can recover it.

---

### 2.2 — SVR with Different Kernels

**What the code does:**
- Fits SVR with three different kernels (`linear`, `poly`, `rbf`) to each of the three datasets (3 × 3 = 9 models total).
- Plots the fitted curves over the scattered data.
- Calculates R² score for each combination.

**What is a kernel?**
A kernel is a mathematical trick that lets SVR fit non-linear curves without explicitly transforming the data. Think of it as the "shape" the model is allowed to fit:

- **linear** — can only fit straight lines. Good for degree-1 data, struggles with curves.
- **poly** (polynomial) — can fit curved shapes. Flexible but can be unpredictable.
- **rbf** (Radial Basis Function) — the most flexible. Fits smooth, complex curves. Works well in many situations.

**What is R²?**
R² is "how much of the variation in y does the model explain?" It ranges from 0 to 1 (sometimes negative for very bad models):
- R² = 1.0 → perfect fit
- R² = 0.0 → the model is no better than just predicting the mean
- R² < 0 → the model is actively worse than the mean

**Why StandardScaler?**
SVR uses distances between data points. If one feature is on a scale of 0–1000 and another is 0–1, the large-scale feature dominates the distance calculation and the model ignores the small-scale one. Scaling all features to have mean 0 and standard deviation 1 makes the comparison fair.

---

### 2.3 — Hyperparameter Tuning via Grid Search

**What the code does:**
- Runs a grid search over hyperparameters for SVR on the cubic dataset.
- Tests every combination of: `kernel` (poly or rbf), `C` (1 or 100), `epsilon` (0.1 or 0.5).
- Uses 5-fold cross-validation to evaluate each combination.
- Reports the best combination and a full results table.

**What is a hyperparameter?**
A regular parameter (like the weights `W_` in softmax regression) is learned from data. A *hyperparameter* is a setting you choose *before* training — it controls how the model learns, not what it learns. Examples:
- `C` — regularisation strength. High C = the model tries hard to fit every point. Low C = the model accepts more errors in exchange for a simpler curve.
- `epsilon` — tube width. Larger epsilon = a wider tube = more tolerance for errors.
- `kernel` — the shape the model is allowed to fit (see above).

**What is cross-validation?**
Instead of training once and checking on one validation set, cross-validation splits the training data into 5 parts ("folds"). It trains 5 times, each time using a different fold as the validation set and the remaining 4 for training. The score is averaged across all 5. This gives a more reliable estimate of how well the hyperparameters work.

**How many models get trained?**
2 kernels × 2 C values × 2 epsilon values = 8 combinations. Each combination is evaluated with 5-fold CV = 8 × 5 = **40 models** total. Then the best combination is refitted on the full training set (`refit=True`).

**What does `svr__C` mean?**
When using a pipeline (StandardScaler + SVR bundled together), sklearn needs to know which step a parameter belongs to. The double underscore separates the step name from the parameter name. `svr__C` means "the `C` parameter of the `SVR` step".

---

### 2.4 — Evaluating the Best Model

**What the code does:**
- Takes the best model found by the grid search (already refitted on full training data).
- Scores it on the held-out **test set** — data it has never seen.
- Plots the fitted curve over both training and test data.

**Plain English:**
This is the moment of truth. During grid search we were tuning settings using the training data (even with cross-validation, the best settings were chosen based on that data). The test set is completely independent — the model has never influenced its selection in any way. The test R² tells us how well the model actually generalises to new data.

A good result is when the CV R² (from training) and test R² are close to each other. A big drop would suggest overfitting.

---

## Key Concepts Summary

| Concept | Plain English |
|---|---|
| **Epoch** | One complete pass through all training data |
| **Mini-batch** | A small chunk of training data used for one weight update |
| **Loss** | A number measuring how wrong the model is — lower is better |
| **Gradient** | A direction indicator telling us how to nudge weights to reduce loss |
| **Learning rate** | How big each weight update step is |
| **Early stopping** | Quit training when the model stops improving on validation data |
| **Overfitting** | Model memorises training data and fails on new data |
| **Normalisation** | Scaling inputs to a small range so training is stable |
| **One-hot encoding** | Converting a label like '3' into a vector like [0,0,0,1,0,0,0,0,0,0] |
| **Softmax** | Converts raw scores into probabilities that sum to 1 |
| **Cross-entropy** | Measures how far predicted probabilities are from the true label |
| **Kernel (SVR)** | The shape the SVR is allowed to fit (linear, polynomial, rbf) |
| **R²** | How much of the data variance the model explains (1.0 = perfect) |
| **Regularisation (C)** | Controls the trade-off between fitting data and keeping the curve simple |
| **Cross-validation** | Training/evaluating multiple times on different splits for a reliable score |
| **Stratification** | Ensuring each data split has the same class proportions |
| **Pipeline** | Bundling preprocessing (scaling) and the model into one reusable object |
