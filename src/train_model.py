import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from data_processing import load_data, clean_data, drop_unused_columns

df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = clean_data(df)
df = drop_unused_columns(df)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

binary_columns = ["Partner", "Dependents", "PaperlessBilling"]
for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

onehot_columns = ["InternetService", "Contract", "PaymentMethod", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df = pd.get_dummies(df, columns=onehot_columns)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Baseline: Logistic Regression, class_weight="balanced".
# Discarded — worse recall on churn than the final model. See NOTES.md.
# model = LogisticRegression(max_iter=1000, class_weight="balanced")
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# Random Forest, class_weight="balanced".
# Discarded — better accuracy/precision but worse recall on churn. See NOTES.md.
# rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)
# print(classification_report(y_test, y_pred_rf))

# --- Final model: Gradient Boosting. See NOTES.md for full comparison. ---

gb_model = GradientBoostingClassifier(random_state=42)
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
gb_model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred_gb = gb_model.predict(X_test)
print(classification_report(y_test, y_pred_gb))

joblib.dump(gb_model, "model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")