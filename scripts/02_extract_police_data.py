from pathlib import Path
import zipfile

ZIP_FILE = Path("data/raw/police_latest.zip")
EXTRACT_DIR = Path("data/raw/extracted")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Zip file not found: {zip_path}\n"
            "Run scripts/01_download_police_data.py first."
        )

    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {zip_path}")
    print(f"Destination: {extract_to}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print("Extraction complete.")


if __name__ == "__main__":
    extract_zip(ZIP_FILE, EXTRACT_DIR)