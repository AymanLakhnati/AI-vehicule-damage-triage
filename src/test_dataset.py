from pathlib import Path

from torchvision import transforms

from dataset import CarDamageDataset

from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CarDamageDataset(
    csv_path=Path("data/splits/train_labels.csv"),
    images_dir=Path("data/raw/car-damage-dataset/images"),
    transform=transform,
)

print(f"Dataset size: {len(dataset)}")

image, label = dataset[0]

print(f"Image shape: {image.shape}")
print(f"Label: {label}")
print(f"Image dtype: {image.dtype}")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

images, labels = next(iter(loader))

print(f"Batch image shape: {images.shape}")
print(f"Batch label shape: {labels.shape}")
print(f"Batch labels: {labels}")