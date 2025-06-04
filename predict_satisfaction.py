import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(path: str) -> pd.DataFrame:
    """Load the hotel satisfaction dataset."""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """Encode categorical columns and split features/labels."""
    df = df.copy()
    categorical = ["Gender", "purpose_of_travel", "Type of Travel", "Type Of Booking"]
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    X = df.drop("satisfaction", axis=1)
    y = (df["satisfaction"] == "satisfied").astype(int)
    return X, y


def train_model(X, y):
    """Train a RandomForest classifier and return the fitted model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    return model


def main():
    data_path = "data/processed/Europe Hotel Booking Satisfaction Score.csv"
    df = load_data(data_path)
    print("Satisfaction distribution:\n", df["satisfaction"].value_counts())
    X, y = prepare_features(df)
    train_model(X, y)


if __name__ == "__main__":
    main()
