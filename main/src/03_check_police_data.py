from pathlib import Path
import pandas as pd

EXTRACT_DIR = Path("../data/raw/extracted")


def check_data(folder: Path) -> None:
    if not folder.exists():
        raise FileNotFoundError(
            f"Extracted folder not found: {folder}\n"
            "Run main/02_extract_police_data.py first."
        )

    csv_files = list(folder.rglob("*.csv"))

    print(f"Number of CSV files found: {len(csv_files)}")

    if not csv_files:
        print("No CSV files found.")
        return

    print("\nFirst 10 files:")
    for file in csv_files[:10]:
        print(file)

    print("\nLoading first CSV file as sample...")
    sample_file = csv_files[0]
    df = pd.read_csv(sample_file, low_memory=False)

    print(f"\nSample file: {sample_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    check_data(EXTRACT_DIR)