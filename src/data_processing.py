import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

def drop_unused_columns(df):
    df = df.drop(columns=["customerID", "TotalCharges", "gender", "PhoneService", "MultipleLines"])
    return df

def main():
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)
    df = drop_unused_columns(df)
    return df

if __name__ == "__main__":
    df = main()
    print(df.head())