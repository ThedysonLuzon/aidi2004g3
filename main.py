# main.py
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def load_penguins_data():
    """Load the penguins dataset from seaborn."""
    try:
        penguins = sns.load_dataset("penguins")
        return penguins
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def main():
    # Load the dataset
    df = load_penguins_data()

    if df is not None:
        # Display basic information
        print("Penguins Dataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nOriginal dataset shape:", df.shape)

    # one-hot encode categorical inputs
        X_train = pd.get_dummies(
            train_df[feature_cols + ["island", "sex"]],
            drop_first=True
        )
        X_test = pd.get_dummies(
            test_df[feature_cols + ["island", "sex"]],
            drop_first=True
        )
        # align train/test columns (in case some category wasn't present)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

        # Label-encode the target
        le = LabelEncoder()
        y_train = le.fit_transform(train_df["species"])
        y_test = le.transform(test_df["species"])

        # === Person C: Fit the model on the training data ===
        model.fit(X_train, y_train)
        print("\nModel training complete.")

        # Quick train score
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy:     {test_score:.3f}")

    else:
        print("Failed to load the penguins dataset.")

if __name__ == "__main__":
    main()