# CITS5508: Assignment 1
**Semester 1, 2026**
**Worth: 20% | Due: 11:59 pm Friday 17th April 2026**

---

## 1 Outline

This assignment is divided into two parts: a **classification** and a **regression** task.

Treat your submission as a **report**. Do not expect full marks for one-word answers — explain your thoughts in detail.

---

## 2 Submission

- Submit a single **Jupyter Notebook** (`.ipynb`) via LMS.
- The notebook must include all code, outputs, descriptions, and comments.
  - Use **Markdown cells** to explain your code and methodology.
  - Divide into sections/subsections with numbered headings.
  - Reference `sample.ipynb` (Week 2 on LMS) for structure guidance.
- The notebook must run **without errors** in the CITS5508 Anaconda environment (Google Colab produces identical outputs).

### 2.1 AI Use Policy

Tier 2: AI assistance or collaboration (UWA AI Assessment Guidelines).

- You **may** use AI tools to assist, but **no AI-generated content** is allowed in the final submission.
- All descriptions, code, and outputs must be **in your own words**.
- You must be able to **justify and explain** everything you submit.

---

## 3 Part 1: Softmax Regression

Implement a **Softmax Regression classifier from scratch** using gradient descent, and apply it to the **MNIST dataset**.

### Background

Softmax Regression generalises logistic regression to the multiclass case. The estimated probability of $x$ belonging to class $k$ is:

$$\hat{p}_k = \frac{\exp(s_k(x))}{\sum_{j=1}^{K} \exp(s_j(x))}$$

where $K$ is the total number of classes and $s_k(x) = (\theta^{(k)})^\top x$.

The **cross-entropy loss** cost function is:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log \hat{p}_k^{(i)}$$

where $y_k^{(i)}$ is the target probability that the $i$-th instance belongs to class $k$.

The gradient with respect to $\theta^{(k)}$ is:

$$\frac{\partial J}{\partial \theta^{(k)}} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{p}_k^{(i)} - y_k^{(i)} \right) x^{(i)}$$

### Tasks

**1.** Load the MNIST dataset from OpenML:

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
```

**2.** Display some examples from the dataset. Describe what the features and labels represent.

**3.** Split the dataset into **training (70%)**, **validation (15%)**, and **test (15%)** sets.

> Hint: use `sklearn.model_selection.train_test_split`

- **(a)** Display the class distribution in each split.
- **(b)** Do you need to stratify your split? Why or why not?

**4.** Implement softmax regression with gradient descent using **vectorised NumPy code**. Implement **early stopping** based on the validation set. Do **not** use scikit-learn — only NumPy and basic Python.

- **(a)** Plot training and validation loss over time.
- **(b)** What do you observe? Is this what you expected?

**5.** Fit a `sklearn.linear_model.LogisticRegression` model (no regularisation). Compare it to your implementation.

- **(a)** Display a **confusion matrix** for each model.
- **(b)** Show appropriate **evaluation metrics** for both models.
- **(c)** Comment on the results. Are they what you expected? Why or why not?

**6.** Comment on how the performance of your softmax regression model compares to the scikit-learn logistic regression model.

---

## 4 Part 2: Support Vector Machine Regression

Investigate how different **kernels** affect the performance of **SVM regression** on synthetic datasets.

### Background

You will generate toy datasets using polynomial functions, then fit `sklearn.svm.SVR` models with different kernels.

### Dataset Generation

Generate random $n$-degree polynomial datasets of the form:

$$y = \sum_{k=0}^{n} a_k x^k + \epsilon$$

where:
- $x \in [-3, 3]$
- $a_k \sim \mathcal{U}(0, 1)$ (uniform distribution)
- $\epsilon \sim \mathcal{N}(0, 1)$ (Gaussian noise)

### Tasks

**1.** Write a function `generate_polynomial_dataset(degree, n_samples)` that generates the dataset above. Plot examples for degree = 1, 2, and 3 to verify it works.

**2.** Generate **three datasets**: linear (degree=1), quadratic (degree=2), and cubic (degree=3), each with `n_samples=1000`. Create a **training and test split** for each.

**3.** Fit a `sklearn.svm.SVR` model to each dataset using:
- `kernel='linear'`
- `kernel='poly'`
- `kernel='rbf'`

- **(a)** Plot the fitted regression function for each kernel × dataset combination.
- **(b)** Comment on the results. Are they what you expected?

**4.** Perform **grid-search with 5-fold cross-validation** on at least **3 hyperparameters** (including `kernel`) with **2 values each** ($2^3 = 8$ total combinations) on the **cubic dataset**.

- **(a)** What hyperparameters did you choose? How many models were fit in total?
- **(b)** Present results for the optimal hyperparameter combination.

**5.** Evaluate the optimal model from Step 4 on the **test set**. Present your results.

---

## Summary Checklist

### Part 1
- [ ] Load and display MNIST examples
- [ ] Train/validation/test split with class distribution plots
- [ ] Softmax regression from scratch (NumPy only) with early stopping
- [ ] Training/validation loss plot
- [ ] scikit-learn LogisticRegression comparison
- [ ] Confusion matrices and evaluation metrics for both models
- [ ] Written commentary throughout

### Part 2
- [ ] Polynomial dataset generator function with demo plots
- [ ] Three datasets (linear, quadratic, cubic) with train/test splits
- [ ] SVR fitted with linear, poly, and rbf kernels — plots for all combinations
- [ ] Grid search with 5-fold CV on cubic dataset (≥3 hyperparams, 8 combinations)
- [ ] Best model evaluated on test set
- [ ] Written commentary throughout
