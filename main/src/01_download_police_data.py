from pathlib import Path
import requests

# Police.uk latest monthly archive
URL = "https://data.police.uk/data/archive/latest.zip"

# Save location
RAW_DIR = Path("../data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "police_latest.zip"


def download_file(url: str, output_path: Path) -> None:
    print("Starting download...")
    print(f"URL: {url}")
    print(f"Saving to: {output_path}")

    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        print(f"\rDownloaded: {percent:.2f}%", end="")

    print("\nDownload complete.")
    print(f"File saved at: {output_path}")


if __name__ == "__main__":
    download_file(URL, OUTPUT_FILE)