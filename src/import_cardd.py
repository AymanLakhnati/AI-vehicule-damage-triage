from pathlib import Path
import zipfile


EXPECTED_ITEMS = [
    "annotations",
    "train2017",
    "val2017",
    "test2017",
]


def find_dataset_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "raw" / "cardd" / "CarDD_release" / "CarDD_COCO",
        project_root / "data" / "raw" / "cardd" / "CarDD_COCO",
    ]

    for candidate in candidates:
        if candidate.exists() and all((candidate / item).exists() for item in EXPECTED_ITEMS):
            return candidate

    archive_candidates = [
        project_root / "data" / "raw" / "cardd" / "CarDD_release" / "*.zip",
        project_root / "data" / "raw" / "cardd" / "*.zip",
        project_root / "data" / "raw" / "*.zip",
    ]

    archive_paths = []
    for pattern in archive_candidates:
        archive_paths.extend(pattern.parent.glob(pattern.name))

    for archive_path in sorted(archive_paths):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(archive_path.parent)

        for candidate in candidates:
            if candidate.exists() and all((candidate / item).exists() for item in EXPECTED_ITEMS):
                return candidate

    raise FileNotFoundError(
        "Could not find a valid CarDD COCO dataset directory or archive under data/raw/cardd"
    )


def ensure_cardd_layout(project_root: Path) -> Path:
    dataset_dir = find_dataset_dir(project_root)
    missing = [item for item in EXPECTED_ITEMS if not (dataset_dir / item).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected CarDD dataset folders: {', '.join(missing)}")
    return dataset_dir


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = ensure_cardd_layout(project_root)
    print(f"CarDD dataset ready at: {dataset_dir}")


if __name__ == "__main__":
    main()
