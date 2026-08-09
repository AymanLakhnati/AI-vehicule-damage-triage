import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    project_root = Path(__file__).resolve().parents[1]
    target_dir = project_root / "data" / "raw" / "car-damage-dataset"
    target_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "vinayjose/car-damage-dataset",
        path=str(target_dir),
        unzip=True,
        quiet=False,
    )

    print(f"Dataset downloaded and extracted to: {target_dir.resolve()}")


if __name__ == "__main__":
    main()
