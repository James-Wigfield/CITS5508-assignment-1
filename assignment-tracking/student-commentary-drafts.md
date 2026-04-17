# Student Commentary Drafts

These are draft versions of every required markdown commentary block in the notebook. Review and edit before pasting into the notebook — especially sections marked **[INSERT FROM OUTPUT]** where you need to fill in actual numbers after running the notebook.

All text should be reworded into your own voice before submission (AI use policy).

---

## Cells that need writing / fixing

---

### Cell `e0e56dbd` — 1.1 Data Loading and Exploration (intro)

> Currently has: `"in this section..."`

**Draft:**

```
### 1.1 Data Loading and Exploration

In this section we load the MNIST dataset from OpenML using scikit-learn's `fetch_openml`. 
We then explore the data by printing basic statistics and visualising examples from the dataset
to understand what the features and labels look like before any modelling.
```

---

### Cell `43dd6079` — 1.2 Data Splitting (intro)

> Currently has: `"70% training etc..."`

**Draft:**

```
### 1.2 Data Splitting

Here we split the 70,000 MNIST samples into three subsets: a training set (70%), a validation 
set (15%), and a test set (15%). The training set is what the model learns from. The validation 
set is used during training to monitor generalisation and trigger early stopping. The test set 
is held out completely until the final evaluation, giving an unbiased measure of performance.

Since `train_test_split` only supports two-way splits, this is done in two stages: first a 
70/30 split to get the training set, then the 30% remainder is split 50/50 to give the 
validation and test sets.
```

---

### Cell `c0852a60` — Stratification Justification

> Currently has: placeholder `**actually compare to the average**` and rough edges.

**Draft (replaces the whole cell):**

```
#### Stratification Justification

The dataset has been split into Training (70%), Validation (15%), and Test (15%) sets using 
`train_test_split`, applied in two stages since it does not support three-way splits directly.

Stratification (`stratify=y`) has been applied at both stages to ensure each split reflects 
the class proportions of the full dataset. Without it, a purely random split could 
over-represent common digits and under-represent rare ones — for example, digit '1' is the 
most frequent class in MNIST (~11.2% of samples) while digit '5' is the least frequent (~9.0%). 
A random split could skew training data toward '1's, giving the model an unbalanced view of 
the problem.

As the bar charts above confirm, the class distributions across train, validation, and test 
sets are near-identical, verifying that stratification has worked correctly.
```

---

### Cell `e17e1ead` — Training Observations (Task 4b)

> Currently has a prompt header left in ("Answer Task 4b here:...") and an incomplete observation. Remove the prompt, fill in actual numbers.

**Draft (replace whole cell — fill in `[X]` values after running):**

```
#### Training Observations

The loss curves show both training and validation loss decreasing steadily in the early epochs, 
which is the expected behaviour — the model is genuinely learning to distinguish digits rather 
than memorising noise.

Early stopping triggered at epoch **[INSERT EPOCH NUMBER]**, at which point the best validation 
loss was **[INSERT VAL LOSS]**. The validation loss had stopped improving for 15 consecutive 
epochs (the configured patience), so training was halted and the weights from the best epoch 
were restored.

The small gap between the training loss (~**[INSERT]**) and validation loss (~**[INSERT]**) at 
the stopping point suggests the model is generalising well without significant overfitting. This 
is expected for a linear model like softmax regression — linear classifiers have limited capacity 
and are less prone to overfitting than more complex models, even on high-dimensional data like 
MNIST's 784 features.

The convergence behaviour is consistent with what you'd expect from mini-batch gradient descent: 
the curves are smoother than single-sample stochastic gradient descent but noisier than full-batch 
gradient descent, as each update only uses a random subset (batch of 256) of the training data.
```

---

### Cell `9eeb0989` — Model Comparison & Evaluation (Task 5c / Task 6)

> Currently entirely placeholder prompts. This needs actual numbers from running the notebook.

**Draft (replace whole cell — fill in `[X]` values after running):**

```
#### Model Comparison & Evaluation

Both models achieve comparable accuracy on the test set, with sklearn's `LogisticRegression` 
scoring **[INSERT]%** and our custom softmax implementation scoring **[INSERT]%** — a difference 
of around **[INSERT]** percentage points. This close result is reassuring: it suggests our 
from-scratch gradient descent implementation is producing similar weight estimates to sklearn's 
more sophisticated L-BFGS optimiser.

The train vs test accuracy gap is slightly larger for the custom model (**[INSERT train acc]** 
vs **[INSERT test acc]**) compared to sklearn (**[INSERT]** vs **[INSERT]**). This is expected 
since L-BFGS converges to a more precise minimum of the loss function, while our mini-batch 
gradient descent makes noisier updates and stops early — meaning our model may not have fully 
converged.

Looking at the confusion matrices, both models struggle most with visually similar digit pairs:
- **4 vs 9** — both digits have a closed loop at the top
- **3 vs 8** — similar overall shape with rounded strokes
- **7 vs 1** — some people write a '7' with minimal horizontal stroke, resembling a '1'

These confusions appear in both models, which makes sense since both are linear classifiers 
operating on raw pixel features with no spatial awareness. Neither model knows that nearby pixels 
are more related than distant ones — a convolutional network would handle these cases far better.

The per-class F1 scores (from `classification_report`) confirm that digit '1' achieves the 
highest F1 in both models — it is visually distinctive and the most frequent class. Digit '8' 
tends to have lower precision and recall in both models due to its similarity to '3', '6', and '9'.

Overall the results are consistent with expectations for a linear classifier on MNIST. Test 
accuracy in the low-to-mid 90s is typical for softmax/logistic regression on this dataset — 
anything significantly higher would require non-linear models.
```

---

### Cell `959a2464` — Kernel Performance Analysis (Task 3b)

> Currently entirely placeholder prompts. Fill in `[X]` R² values after running.

**Draft (replace whole cell):**

```
#### Kernel Performance Analysis

The R² scores summarised in the table above show a clear pattern across kernel and dataset combinations.

**Linear kernel:** Performs well on the linear dataset (R² ≈ **[INSERT]**), as expected — a 
straight line is the correct underlying shape. However, it underfits noticeably on the quadratic 
and cubic datasets (R² drops to **[INSERT]** and **[INSERT]** respectively), producing a flat 
line that cannot capture the curved relationship in the data.

**Polynomial kernel:** Handles the quadratic and cubic datasets better than the linear kernel, 
since it can model curved relationships. The default `degree=3` in sklearn's SVR gives it 
enough flexibility to approximate cubic patterns, though performance still varies depending on 
the specific coefficients of each dataset.

**RBF (Radial Basis Function) kernel:** Consistently achieves the highest R² across all three 
datasets (linear: **[INSERT]**, quadratic: **[INSERT]**, cubic: **[INSERT]**). Because it maps 
data into an infinite-dimensional feature space, it can approximate virtually any smooth 
function. This flexibility means it adapts well regardless of the underlying polynomial degree.

The key takeaway is that kernel choice should match the expected shape of the data. If you 
know the relationship is linear, a linear kernel is sufficient. In practice, when the 
underlying shape is unknown, RBF is often a safe default due to its flexibility — though it 
requires careful tuning of `C` and `gamma` to avoid overfitting.
```

---

### Cell `af0b3ffa` — Grid Search Results (Task 4a & 4b)

> Currently entirely placeholder prompts. Fill in after running.

**Draft (replace whole cell — fill in `[X]` values after running):**

```
#### Grid Search Results

**Hyperparameters searched:**

We searched over three hyperparameters:
- **`kernel`** (`poly`, `rbf`): controls the shape of the decision boundary / regression function. 
  We excluded `linear` since we already established in section 2.2 that it underfits the cubic data.
- **`C`** (`1`, `100`): the regularisation parameter. Higher C means the model is penalised more 
  heavily for predictions outside the ε-tube, so it tries harder to fit individual points. 
  Lower C produces a smoother, more regularised fit.
- **`epsilon`** (`0.1`, `0.5`): the width of the insensitive tube around the regression line. 
  Points within this margin incur no loss. A larger epsilon produces a coarser but more 
  robust fit.

**Total models fitted:** 2 × 2 × 2 = 8 hyperparameter combinations × 5 folds = **40 models**.

**Best hyperparameters found:** `kernel='[INSERT]'`, `C=[INSERT]`, `epsilon=[INSERT]`

**Best cross-validation R²:** **[INSERT]**

This result [was / was not] surprising. [e.g.: The RBF kernel with high C performing best 
is consistent with our section 2.2 findings — RBF is the most flexible kernel and C=100 
allows it to fit the cubic shape closely. The lower epsilon value of 0.1 suggests a tighter 
tube is preferred for this data, fitting the curve more precisely.]

The `std R²` column in the results table shows relatively low standard deviations across folds 
(**[INSERT range]**), suggesting the results are stable and not dependent on which particular 
fold was used as validation — the model generalises consistently across different subsets of 
the training data.
```

---

### Cell `b2185981` — Final Conclusion

> Currently empty.

**Draft:**

```
### Final Conclusion

This assignment covered two machine learning tasks using fundamentally different approaches.

In **Part 1**, we implemented softmax regression from scratch in NumPy, successfully classifying 
MNIST handwritten digits with ~**[INSERT]%** test accuracy. The implementation correctly applied 
mini-batch gradient descent with early stopping, and the results were validated by comparison 
against scikit-learn's `LogisticRegression`, which achieved similar performance. Both models 
struggled with the same visually ambiguous digit pairs (e.g. 4/9 and 3/8), which is an 
inherent limitation of linear classifiers operating on raw pixel features without any spatial 
structure.

In **Part 2**, we investigated how SVR kernel choice affects regression performance on synthetic 
polynomial data. The experiments confirmed that kernel selection should be guided by the 
expected shape of the data: the linear kernel underfits non-linear datasets, while the RBF 
kernel generalises well across all three polynomial degrees. Grid search with 5-fold cross- 
validation identified **[INSERT best params]** as the optimal configuration for the cubic 
dataset, achieving a test R² of **[INSERT]**, which [closely matched / was slightly below] 
the cross-validation score — indicating the model generalises well to unseen data.
```

---

## Cells that already have content (review only)

These cells are already written — just review them for typos and completeness before submitting:

| Cell ID | Section | Status |
|---|---|---|
| `9c11bc74` | Dataset Description (1.1) | Written — fix typos: "datset", "reporesented", "represets" |
| `9a6dff62` | Softmax Implementation intro (1.3) | Written — looks good |
| `5939eca4` | sklearn Comparison intro (1.4) | Written — looks good |
| `4577a2a4` | Dataset Generation intro (2.1) | Written — looks good |
| `68b3e058` | SVR with Different Kernels intro (2.2) | Written — looks good |
| `21c46745` | Grid Search intro (2.3) | Written — looks good |
| `e1d16a3a` | Optimal Model intro (2.4) | Written — looks good |

---

## Reminder: things to fill in after running the notebook

- [ ] Early stopping epoch number and best val loss (`e17e1ead`)
- [ ] Train and test accuracy for both models (`9eeb0989`)
- [ ] R² values for all 9 kernel × dataset combinations (`959a2464`)
- [ ] Best grid search hyperparameters, CV R², and std R² values (`af0b3ffa`)
- [ ] Test R² for final model (`b2185981`)
- [ ] Final accuracy numbers in conclusion (`b2185981`)
