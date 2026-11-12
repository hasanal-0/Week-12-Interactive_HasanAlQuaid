"""
AIPI 510: Week 12 Interactive
Hasan Al-Quaid 

Simple, reproducible ML pipeline: logistic regression on the breast cancer dataset.

Split: 80% train / 10% validation / 10% test.
  train      -> fit the model
  validation -> choose the hyperparameter C
  test       -> report final performance, once

"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# fixed seed for reporduciblitity, If you use the same splits, will get same results every run
SEED = 11  

# load breast cancer data
X, y = load_breast_cancer(return_X_y=True)

# splits 80/10/10(according to chapter 5 in unit 1), stratified so every set keeps the same malignant(cancerours)/benign(not cancerous) ratio
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size = 0.2, 
    stratify = y, 
    random_state = SEED
    )
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp, 
    test_size = 0.5, 
    stratify = y_temp, 
    random_state = SEED
    )

print(f"Train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}\n")


def make_model(C):
    """
    Build a scaler and logistic regression model as one pipeline, so the scaler only learns from training data

    Parameters:
        C (float): Regularization strength for logistic regression; smaller values mean stronger regularization (a simpler model)

    Returns:
        Pipeline: An unfitted scikit-learn pipeline that standardizes the features, then applies logistic regression
    """
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=1000, random_state=SEED))


# use the validation set to choose our hyperparam (train on train, score on validation)
# trying a few values of C and keeping the one that scores best on the validation set
C_values = [0.01, 0.1, 1, 10]
best_C = None
best_val_acc = 0

for C in C_values:
    # train model with this C on the training set
    model = make_model(C)
    model.fit(X_train, y_train)

    # score it on the validation set
    val_predictions = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_predictions)
    print(f"C = {C}: validation accuracy = {val_acc:.4f}")

    # Keep this C if it beats the best so far
    if val_acc > best_val_acc:
        best_C = C
        best_val_acc = val_acc

print(f"\nChosen C: {best_C}")

# train the final model with the chosen C, then evaluate once on the test set
model = make_model(best_C).fit(X_train, y_train)
y_pred = model.predict(X_test)

# show accuracy
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")

# dispaly other useful info about the model
print(f"\nBaseline (always predict benign): {(y_test == 1).mean():.4f}")
print(f"Recall on malignant (cancers caught): {recall_score(y_test, y_pred, pos_label=0):.4f}")
print("Confusion matrix [rows = actual malignant, benign]:")
print(confusion_matrix(y_test, y_pred))
