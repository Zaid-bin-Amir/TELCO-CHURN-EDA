# Naive Baseline: Majority-Class Classifier

Prepared by: Zaid

## Setup

`DummyClassifier(strategy="most_frequent")` from scikit-learn, fitted and evaluated on the full dataset (7,043 customers), as this task is EDA only and train/test splitting comes later. The classifier ignores every feature and always predicts "No Churn". The churn class (Yes) is treated as the positive class.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.7346 (73.46%) |
| Precision (churn) | 0.0000 |
| Recall (churn) | 0.0000 |
| F1 (churn) | 0.0000 |

Confusion matrix:

| | Predicted No | Predicted Yes |
|---|---|---|
| Actual No | 5,174 | 0 |
| Actual Yes | 1,869 | 0 |

The model catches 0 of 1,869 churners. Precision is reported as 0 because it makes no positive predictions at all (the ratio is undefined, so scikit-learn is set to return 0).

## Why accuracy alone is misleading here

A model that does nothing useful scores 73.46% accuracy, simply because 73.46% of customers do not churn. Accuracy rewards getting the majority class right, so with imbalanced data a high number can hide a model that finds no churners at all. Since the business goal is to identify customers who are about to leave, recall and F1 for the churn class tell the real story, and here they are both zero.

For contrast, a classifier that always predicts "Churn" gets accuracy 0.2654, precision 0.2654, recall 1.0000, and F1 0.4194. Its accuracy looks terrible, yet it has a higher F1 than the "always No" model, which shows how much the choice of metric changes the ranking.

## Why this baseline is still useful

This is a meaningful reference point and not a joke:

- Any real model must beat 73.46% accuracy to show it has learned anything beyond the class balance.
- Any real model must produce a churn-class F1 above 0.00, and ideally well above the 0.42 that blind guessing "Churn" achieves.
- If a later model reports around 75% accuracy, this baseline shows that is only about 1.5 points better than doing nothing, so recall and precision for churners must be checked before calling it good.

Later model comparisons should use churn-class precision, recall, and F1, plus PR-AUC or ROC-AUC, with stratified train/test splits.
