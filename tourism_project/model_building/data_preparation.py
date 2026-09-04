from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
ARTIFACTS = ROOT / "artifacts"

def main():
    df = pd.read_csv(DATA_PATH)

    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)

    for col in ["Unnamed: 0", "CustomerID"]:
        if col in df.columns:
            df = df.drop(columns=col)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["ProdTaken"]
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(ARTIFACTS / "train.csv", index=False)
    test_df.to_csv(ARTIFACTS / "test.csv", index=False)

    print(f"Rows before duplicate removal: {before}")
    print(f"Rows after cleaning: {after}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

if __name__ == "__main__":
    main()
