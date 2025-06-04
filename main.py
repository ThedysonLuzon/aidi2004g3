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

        # Create a default xgboost model (Person B)
        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        print("\nDefault XGBoost model instantiated:")
        print(model)

    else:
        print("Failed to load the penguins dataset.")

if __name__ == "__main__":
    main()