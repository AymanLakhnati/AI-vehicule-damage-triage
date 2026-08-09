import json
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from cardd_model import build_cardd_model
from cardd_dataset import CLASSES
from cardd_dataloaders import build_dataloaders

MODEL_PATH = Path('models/cardd_resnet18_finetuned.pth')
THRESHOLDS_PATH = Path('models/cardd_thresholds.json')
OUTPUT_DIR = Path('reports/error_analysis')


def load_thresholds(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_error_image(image_path: Path, output_path: Path, actual, predicted, prob, threshold):
    img = Image.open(image_path).convert('RGB')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_name = output_path.name
    img.save(output_path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    _, _, test_loader = build_dataloaders(batch_size=32, num_workers=0)
    model = build_cardd_model().to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model weights not found: {MODEL_PATH}')
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f'Threshold file not found: {THRESHOLDS_PATH}')

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    thresholds = load_thresholds(THRESHOLDS_PATH)
    threshold_tensor = torch.tensor(
        [
            thresholds['dent'],
            thresholds['scratch'],
            thresholds['crack'],
            thresholds['glass shatter'],
            thresholds['lamp broken'],
            thresholds['tire flat'],
        ],
        device=device,
    )

    # Reconstruct image paths from test loader's dataset
    test_ds = test_loader.dataset
    data_root = Path('data/raw/cardd/CarDD_release/CarDD_COCO/test2017')
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect results per sample
    results = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_ds)):
            image_tensor, target = test_ds[idx]
            image_id = test_ds.image_ids[idx]
            file_name = test_ds.image_id_to_file[image_id]
            image_path = data_root / file_name

            input_tensor = image_tensor.unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().squeeze(0)
            preds = (probs >= threshold_tensor.cpu()).int().cpu()

            results.append({
                'image_id': image_id,
                'file_name': file_name,
                'target': target.numpy(),
                'pred': preds.numpy(),
                'probs': probs.numpy(),
            })

    # Save a few FP/FN cases for each class
    for cls_idx, cls_name in enumerate(CLASSES):
        fp_dir = OUTPUT_DIR / cls_name.replace(' ', '_') / 'false_positives'
        fn_dir = OUTPUT_DIR / cls_name.replace(' ', '_') / 'false_negatives'
        fp_dir.mkdir(parents=True, exist_ok=True)
        fn_dir.mkdir(parents=True, exist_ok=True)

        fp_count = 0
        fn_count = 0
        fp_suffix = 1
        fn_suffix = 1

        for item in results:
            actual = item['target'][cls_idx]
            predicted = item['pred'][cls_idx]
            prob = float(item['probs'][cls_idx])
            threshold = float(threshold_tensor[cls_idx].cpu().item())
            file_name = item['file_name']

            if actual == 0 and predicted == 1 and fp_count < 5:
                out_name = f"{cls_name.replace(' ', '_')}_FP_prob_{prob:.2f}_th_{threshold:.2f}_{fp_suffix:03d}.jpg"
                save_error_image(
                    data_root / file_name,
                    fp_dir / out_name,
                    actual,
                    predicted,
                    prob,
                    threshold,
                )
                fp_count += 1
                fp_suffix += 1

            if actual == 1 and predicted == 0 and fn_count < 5:
                out_name = f"{cls_name.replace(' ', '_')}_FN_prob_{prob:.2f}_th_{threshold:.2f}_{fn_suffix:03d}.jpg"
                save_error_image(
                    data_root / file_name,
                    fn_dir / out_name,
                    actual,
                    predicted,
                    prob,
                    threshold,
                )
                fn_count += 1
                fn_suffix += 1

            if fp_count >= 5 and fn_count >= 5:
                break

        print(f"Saved {fp_count} false positives and {fn_count} false negatives for {cls_name}")


if __name__ == '__main__':
    main()
