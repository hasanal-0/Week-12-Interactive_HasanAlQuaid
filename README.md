# Week 12 Interactive: Simple ML Pipeline

By Hasan Al-Quaid

Description:
\
Logistic regression that classifies breast tumors as malignant or benign, using scikit-learn's built-in Breast Cancer Wisconsin dataset (569 patients, 30 features).

## Results

**Test accuracy = 96.49%** (55 of 57 test patients correct)

| Metric | Value |
|---|---|
| Test accuracy | 0.9649 |
| Baseline (always predict "benign") | 0.6316 |
| Recall on malignant (cancers caught) | 0.9048 (19 of 21) |
| Precision on malignant | 1.0000 (0 false alarms) |
| Validation accuracy (chosen C = 1) | 0.9474 |

Confusion matrix on the test set (rows = actual, columns = predicted):

|  | Predicted malignant | Predicted benign |
|---|---|---|
| **Actually malignant** | 19 | 2 (missed cancers) |
| **Actually benign** | 0 (false alarms) | 36 |

**What this means**
- The model beats the 63% "always benign" baseline by about 33 points, so it learned real patterns.
- Both of its mistakes were **missed cancers**: 2 of 21 malignant tumors were predicted benign. It made no false alarms. For a medical screening tool, missing a cancer is the more costly error, so improving recall on malignant would be the priority for a real system.
- The test set is small (57 patients), so each mistake moves accuracy by about 1.75 points. The 95% confidence interval for accuracy is roughly **88% to 99%**.

## How to reproduce

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python ML.py
```

You should see `Chosen C: 1` and `Test accuracy: 0.9649`.

## Pipeline

1. **Data:** `load_breast_cancer()` from scikit-learn. It's bundled with the library, so everyone gets the same data with no download. Labels: 0 = malignant, 1 = benign.
2. **Split:** 80% train / 10% validation / 10% test (455 / 57 / 57), stratified so every set keeps the same malignant/benign ratio, with a fixed random seed(I used 11). This follows the three-split workflow from Chapter 5 in Unit 1 of the class material.
3. **Model:** `StandardScaler` and `LogisticRegression` in one pipeline, so the scaler learns only from the training data (no data leakage).
4. **Tune on validation:** Try C = 0.01, 0.1, 1, 10. Train each on the training set and score it on the validation set. C = 1 scored best (0.9474).
5. **Test once:** Train the final model with C = 1 and score it once on the test set. Nothing was changed after seeing the test result.

| Set | Size | Used for |
|---|---|---|
| Train | 80% (455) | fit the model |
| Validation | 10% (57) | choose C |
| Test | 10% (57) | report final performance, once |

Validation results:

| C | Validation accuracy |
|---|---|
| 0.01 | 0.9298 |
| 0.1 | 0.9298 |
| **1** | **0.9474** |
| 10 | 0.9298 |

## Why it's reproducible

- A fixed random seed for both splits and the model
- The pinned scikit-learn version in `requirements.txt`
- Data that ships with scikit-learn, so it can't change
- One script that runs the whole pipeline