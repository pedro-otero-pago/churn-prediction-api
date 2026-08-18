## Exploratory Data Analysis (EDA)

I put the visualization code in a separate file, eda.py, instead of
mixing it into data_processing.py. data_processing.py holds only
reusable cleaning logic that the future training script will depend on,
while eda.py is a standalone script anyone can run to regenerate the
exploratory plots (saved as PNGs in reports/) without needing a Jupyter
notebook.

I initially had eda.py call drop_unused_columns before generating the
plots, which broke the correlation heatmap and the demographics grid,
since those need columns (TotalCharges, gender, PhoneService,
MultipleLines) that drop_unused_columns removes. This surfaced an
ordering problem: dropping columns is a *conclusion* of the EDA, not a
step that should happen before it. eda.py now only loads and cleans the
data (load_data + clean_data), and explores the full dataset;
drop_unused_columns is meant to be used later, by the training script,
which needs the already-reduced dataset.

### Cleaning TotalCharges

TotalCharges is stored as text and has 11 blank values, all belonging
to customers with tenure = 0. Since a brand-new customer hasn't been
billed a total yet, I converted these blanks to 0 instead of dropping
the rows or leaving them null — the value has a clear logical
explanation, unlike a genuinely missing field.

### Numeric features: tenure, MonthlyCharges, TotalCharges

Boxplots of tenure and MonthlyCharges by Churn confirmed both are
strongly related to churn: customers who churn have a much lower median
tenure (~10 months vs ~38) and a higher median MonthlyCharges (~80 vs
~65).

A correlation heatmap showed tenure and TotalCharges are correlated at
0.83 — essentially redundant. I decided to drop TotalCharges and keep
tenure instead, since tenure is more interpretable on its own (months
as a customer) and isn't a mix of tenure and price the way TotalCharges
is. MonthlyCharges correlates more moderately with both (0.65 with
TotalCharges, 0.25 with tenure), so it stays as an independent feature.

### Categorical features

Contract, PaymentMethod, and InternetService all showed clear
differences in churn proportion: month-to-month contracts, electronic
check payment, and fiber optic internet are all associated with
noticeably higher churn than their alternatives.

For demographics and phone-related features, gender and PhoneService
showed almost no difference in churn between categories, and
MultipleLines showed only a weak one — all three were dropped.
SeniorCitizen, Partner, and Dependents did show real differences and
were kept.

The six internet-related service columns (OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies) all follow
a similar pattern by shape (since they share the same "No internet
service" category), but each one still describes a genuinely different
service the customer lacks. Unlike TotalCharges, there's no single
clear redundant pair to drop here, so all six were kept.

### Final column decisions

Dropped: customerID, TotalCharges, gender, PhoneService, MultipleLines.
Kept: everything else, including all six internet service columns.

## Preprocessing and encoding

Categorical columns were split into two groups based on cardinality:
binary columns (Partner, Dependents, PaperlessBilling, plus Churn
itself) were mapped directly to 0/1, while columns with three or more
categories (InternetService, Contract, PaymentMethod, and the six
internet service columns) were one-hot encoded with pd.get_dummies.
Label encoding was avoided for the latter group since it would imply a
false ordering between categories that don't have one.

tenure, MonthlyCharges, and SeniorCitizen were left untouched, since
they're already numeric in the source data (SeniorCitizen is already
stored as 0/1).

## Train/test split

Split 80/20 with stratify=y, since Churn is imbalanced (~26.5%/73.5%).
Without stratification, a random split could easily produce a test set
with a meaningfully different churn ratio than the real data, given how
few churn cases exist relative to the total. With ~374 churn cases
still in the test set at this split, the sample is large enough to
evaluate confidently without needing cross-validation for a project at
this scale.

## Baseline model: Logistic Regression

First model trained: LogisticRegression(max_iter=1000), no class
weighting.

Results on the test set:
- Accuracy: 0.80
- Class 0 (no churn): precision 0.84, recall 0.89
- Class 1 (churn): precision 0.64, recall 0.53

The accuracy figure is misleading on its own: a model that always
predicts "no churn" would already score ~0.735 accuracy, given the
class imbalance. The metric that matters most for this business problem
is recall on class 1 (churn) — missing a customer who was actually
going to leave (a false negative) is worse than a false positive
(offering a discount to someone who wasn't leaving), since a false
negative means losing the customer entirely with no chance to
intervene. A recall of 0.53 means the model misses nearly half of the
customers who actually churn, which isn't good enough as a starting
point.

This is a direct consequence of class imbalance: with far more "No"
examples during training, the model defaults to being conservative
about predicting churn. Next step: try class_weight="balanced" to
penalize errors on the minority class more heavily during training.

## Logistic Regression with class_weight="balanced"

Same model, adding class_weight="balanced" to penalize errors on the
minority class (churn) more heavily during training.

Results on the test set:
- Accuracy: 0.74 (down from 0.80)
- Class 0 (no churn): precision 0.90, recall 0.73
- Class 1 (churn): precision 0.51, recall 0.78 (up from 0.53)

This is the expected trade-off: recall on the class that matters most
improved substantially (missing far fewer real churners), at the cost
of more false positives and a lower overall accuracy. Given that a
false negative (losing a customer with no chance to intervene) is worse
for the business than a false positive (offering an unnecessary
discount), this version is preferable despite the lower headline
accuracy — a case where accuracy alone would lead to the wrong
conclusion.

Still, these results feel like a reasonable but improvable starting
point. Next: try Random Forest, which handles non-linear relationships
and interactions between features (e.g. how MonthlyCharges affects
churn differently depending on Contract type) better than logistic
regression.

## Random Forest with class_weight="balanced"

Results on the test set:
- Accuracy: 0.76 (up from 0.74)
- Class 0 (no churn): precision 0.86, recall 0.81
- Class 1 (churn): precision 0.55, recall 0.62, f1 0.62

Comparing the two class_weight="balanced" models:

| Metric                  | Logistic Regression | Random Forest |
|--------------------------|---------------------|----------------|
| Accuracy                | 0.74                | 0.76           |
| Recall (churn)           | 0.78                | 0.62           |
| Precision (churn)        | 0.51                | 0.55           |
| F1 (churn)                | 0.58                | 0.62           |

There's no absolute winner here — it depends on which metric the
business prioritizes. Random Forest improves accuracy, precision, and
F1 for the churn class, but its recall is meaningfully worse (0.62 vs
0.78), meaning more real churners go undetected. Given the earlier
decision that a false negative (losing a customer with no chance to
intervene) is worse than a false positive, Logistic Regression with
class_weight="balanced" remains the better choice for this business
problem, despite Random Forest's better numbers on paper for most other
metrics.

## Gradient Boosting

GradientBoostingClassifier doesn't expose a class_weight parameter like
LogisticRegression and RandomForestClassifier do. To get the same
effect, sample weights were computed manually with
compute_sample_weight(class_weight="balanced", y=y_train) and passed to
.fit() via sample_weight — a reminder that not every scikit-learn model
handles class imbalance the same way.

Results on the test set:
- Accuracy: 0.75
- Class 0 (no churn): precision 0.91, recall 0.74
- Class 1 (churn): precision 0.52, recall 0.79, f1 0.63

## Model comparison and final choice

| Metric            | Logistic Regression | Random Forest | Gradient Boosting |
|-------------------|----------------------|----------------|--------------------|
| Accuracy          | 0.74                 | 0.76           | 0.75               |
| Recall (churn)    | 0.78                 | 0.62           | 0.79               |
| Precision (churn) | 0.51                 | 0.55           | 0.52               |
| F1 (churn)        | 0.58                 | 0.62           | 0.63               |

Gradient Boosting is the clear winner: unlike Random Forest, it doesn't
trade away recall for better accuracy/precision — it slightly improves
recall over Logistic Regression (0.79 vs 0.78) while also improving
precision and F1, with no negative trade-off on the metric that matters
most for this business problem. Chosen as the final model for the API.

## Saving the model

The trained model and the exact list of columns it expects (after
encoding) were saved separately with joblib: model.pkl for the model
itself, model_columns.pkl for the column list. Saving the columns
separately matters because the future API will one-hot encode incoming
customer data on its own, and needs a way to guarantee the result has
the same columns, in the same order, as what the model was trained on
— including columns that might not appear for a given customer (e.g. a
payment method they don't use), which get filled with 0 via
DataFrame.reindex(columns=model_columns, fill_value=0) rather than
causing a missing-column error.

Loading model.pkl with a different scikit-learn version than the one
used to train it raises an InconsistentVersionWarning. To avoid this
when the model runs inside the future Docker container, requirements.txt
needs to pin the exact scikit-learn version used during training, not
just any version.