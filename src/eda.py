import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data, clean_data, drop_unused_columns

df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = clean_data(df)

def plot_tenure_by_churn(df):
    plt.figure()
    sns.boxplot(data=df, x="Churn", y="tenure")
    plt.title("Tenure by churn status")
    plt.savefig("reports/tenure_by_churn.png")
    plt.close()

def plot_monthly_charges_by_churn(df):
    plt.figure()
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly charges by churn status")
    plt.savefig("reports/monthly_charges_by_churn.png")
    plt.close()

def plot_correlation_heatmap(df):
    plt.figure()
    correlation_matrix = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation between numeric features")
    plt.savefig("reports/correlation_heatmap.png")
    plt.close()

def plot_contract_by_churn(df):
    plt.figure()
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Churn count by contract type")
    plt.savefig("reports/contract_by_churn.png")
    plt.close()

def plot_payment_method_by_churn(df):
    plt.figure()
    sns.countplot(data=df, x="PaymentMethod", hue="Churn")
    plt.title("Churn count by payment method")
    plt.xticks(rotation=45)
    plt.savefig("reports/payment_method_by_churn.png")
    plt.close()

def plot_internet_service_by_churn(df):
    plt.figure()
    sns.countplot(data=df, x="InternetService", hue="Churn")
    plt.title("Churn count by internet service")
    plt.xticks(rotation=45)
    plt.savefig("reports/internet_service_by_churn.png")
    plt.close()

def plot_categorical_grid(df, columns, filename):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.countplot(data=df, x=col, hue="Churn", ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"reports/{filename}.png")
    plt.close()

plot_tenure_by_churn(df)
plot_monthly_charges_by_churn(df)
plot_correlation_heatmap(df)
plot_contract_by_churn(df)
plot_payment_method_by_churn(df)
plot_internet_service_by_churn(df)
plot_categorical_grid(df, ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"], "services_by_churn")
plot_categorical_grid(df, ["SeniorCitizen", "Partner", "Dependents", "gender", "PhoneService", "MultipleLines"], "demographics_by_churn")