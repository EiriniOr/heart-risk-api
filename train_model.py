import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # 1. Load heart disease data
    df = pd.read_csv("heart.csv")

    # 2. Select a few simple, meaningful features
    feature_cols = ["age", "sex", "trestbps", "chol", "thalach", "exang"]
    X = df[feature_cols]
    y = df["target"]  # 1 = has heart disease, 0 = no heart disease

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

    # 5. Save model
    joblib.dump(model, "model.pkl")
    print("Saved model to model.pkl")

if __name__ == "__main__":
    main()