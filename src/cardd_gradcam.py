import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from cardd_model import build_cardd_model
from cardd_dataset import CarDDMultiLabelDataset, CLASSES

MODEL_PATH = Path('models/cardd_resnet18_finetuned.pth')
THRESHOLDS_PATH = Path('models/cardd_thresholds.json')
DATA_ROOT = Path('data/raw/cardd/CarDD_release/CarDD_COCO')
OUTPUT_ROOT = Path('reports/gradcam')

EVALUATION_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
VISUAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
])

SAMPLE_COUNTS = {
    'crack': {'tp': 3, 'fp': 3, 'fn': 3},
    'dent': {'tp': 3, 'fp': 3, 'fn': 3},
    'scratch': {'tp': 3, 'fp': 3, 'fn': 3},
    'glass shatter': {'tp': 3, 'fp': 3, 'fn': 3},
    'lamp broken': {'tp': 3, 'fp': 3, 'fn': 3},
    'tire flat': {'tp': 3, 'fp': 3, 'fn': 3},
}


def load_thresholds(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def compute_gradcam(model, input_tensor, class_idx):
    gradients = []
    activations = []

    def forward_hook(module, input_, output):
        activations.append(output)

        def save_grad(grad):
            gradients.append(grad)

        output.register_hook(save_grad)

    target_layer = model.layer4[-1]
    hook_handle = target_layer.register_forward_hook(forward_hook)

    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    output = model(input_tensor)
    score = output[0, class_idx]
    model.zero_grad()
    score.backward(retain_graph=False)

    hook_handle.remove()

    if not activations or not gradients:
        raise RuntimeError('Grad-CAM hooks did not capture activations or gradients.')

    activation = activations[0].detach().cpu().squeeze(0)
    gradient = gradients[0].detach().cpu().squeeze(0)

    weights = gradient.mean(dim=(1, 2))
    cam = torch.relu((weights.view(-1, 1, 1) * activation).sum(dim=0))
    cam = cam.numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)

    cam = Image.fromarray(np.uint8(cam * 255), mode='L')
    cam = cam.resize((224, 224), resample=Image.BILINEAR)
    return np.array(cam) / 255.0


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap('jet')
    colored_heatmap = cmap(heatmap)[:, :, :3]
    overlay = image.astype(np.float32) / 255.0 * 0.5 + colored_heatmap * 0.5
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def save_gradcam_figure(
    image_path: Path,
    output_path: Path,
    label_names,
    pred_names,
    class_name,
    probability,
    threshold,
    prediction_text,
    heatmap,
):
    pil_image = Image.open(image_path).convert('RGB')
    pil_image = VISUAL_TRANSFORM(pil_image)
    image_np = np.array(pil_image)
    overlay = overlay_heatmap(image_np, heatmap)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_np)
    axes[0].axis('off')
    axes[0].set_title('Original')

    axes[1].imshow(overlay)
    axes[1].axis('off')
    axes[1].set_title('Grad-CAM Overlay')

    title = (
        f"Class: {class_name} | Ground truth: {', '.join(label_names) or 'none'}\n"
        f"Predicted: {', '.join(pred_names) or 'none'} | {prediction_text}\n"
        f"Probability: {probability:.2f} | Threshold: {threshold:.2f}"
    )
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_sample_sets(model, dataset, threshold_tensor, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, target = dataset[idx]
            image = image_tensor.unsqueeze(0).to(device)
            logits = model(image)
            probs = torch.sigmoid(logits).cpu().squeeze(0)
            preds = (probs >= threshold_tensor.cpu()).int().cpu()
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(target)

    all_probs = torch.stack(all_probs)
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    sample_indices = {
        class_name: {'tp': [], 'fp': [], 'fn': []}
        for class_name in CLASSES
    }

    for idx in range(len(dataset)):
        target = all_targets[idx]
        pred = all_preds[idx]
        for class_idx, class_name in enumerate(CLASSES):
            if len(sample_indices[class_name]['tp']) < SAMPLE_COUNTS[class_name]['tp'] and target[class_idx] == 1 and pred[class_idx] == 1:
                sample_indices[class_name]['tp'].append(idx)
            if len(sample_indices[class_name]['fp']) < SAMPLE_COUNTS[class_name]['fp'] and target[class_idx] == 0 and pred[class_idx] == 1:
                sample_indices[class_name]['fp'].append(idx)
            if len(sample_indices[class_name]['fn']) < SAMPLE_COUNTS[class_name]['fn'] and target[class_idx] == 1 and pred[class_idx] == 0:
                sample_indices[class_name]['fn'].append(idx)

    return sample_indices, all_probs, all_preds, all_targets


def names_from_tensor(binary_tensor):
    return [CLASSES[i] for i, value in enumerate(binary_tensor.tolist()) if value == 1]


def run_gradcam_for_samples(
    model,
    dataset,
    sample_indices,
    thresholds,
    all_probs,
    all_preds,
    all_targets,
    device,
):
    for class_idx, class_name in enumerate(CLASSES):
        if class_name not in sample_indices:
            continue
        out_dir = OUTPUT_ROOT / class_name.replace(' ', '_')
        for category in ['tp', 'fp', 'fn']:
            category_dir = out_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            for example_idx, sample_idx in enumerate(sample_indices[class_name][category], start=1):
                image_id = dataset.image_ids[sample_idx]
                file_name = dataset.image_id_to_file[image_id]
                image_path = DATA_ROOT / 'test2017' / file_name

                original = Image.open(image_path).convert('RGB')
                input_tensor = EVALUATION_TRANSFORM(original).unsqueeze(0).to(device)

                probability = float(all_probs[sample_idx, class_idx].item())
                threshold = float(thresholds[class_name])
                prediction_text = 'True Positive' if category == 'tp' else 'False Positive' if category == 'fp' else 'False Negative'

                heatmap = compute_gradcam(model, input_tensor, class_idx)
                gt_names = names_from_tensor(all_targets[sample_idx])
                pred_names = names_from_tensor(all_preds[sample_idx])

                file_name_out = (
                    f"{class_name.replace(' ', '_')}_{category}_{example_idx:02d}_prob_{probability:.2f}_th_{threshold:.2f}.png"
                )
                output_path = category_dir / file_name_out
                save_gradcam_figure(
                    image_path=image_path,
                    output_path=output_path,
                    label_names=gt_names,
                    pred_names=pred_names,
                    class_name=class_name,
                    probability=probability,
                    threshold=threshold,
                    prediction_text=prediction_text,
                    heatmap=heatmap,
                )
                print(f"Saved Grad-CAM for {class_name} {category} example {example_idx} -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for CarDD test examples.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run model on.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print('Using device:', device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model weights not found: {MODEL_PATH}')
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f'Threshold file not found: {THRESHOLDS_PATH}')

    with THRESHOLDS_PATH.open('r', encoding='utf-8') as f:
        thresholds = json.load(f)

    model = build_cardd_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_dataset = CarDDMultiLabelDataset(
        DATA_ROOT / 'annotations' / 'instances_test2017.json',
        DATA_ROOT / 'test2017',
        transform=EVALUATION_TRANSFORM,
    )

    threshold_tensor = torch.tensor(
        [thresholds[cls] for cls in CLASSES],
        device=device,
    )

    sample_indices, all_probs, all_preds, all_targets = build_sample_sets(
        model,
        test_dataset,
        threshold_tensor,
        device,
    )

    for class_name in CLASSES:
        counts = sample_indices[class_name]
        print(f"{class_name}: TP={len(counts['tp'])}, FP={len(counts['fp'])}, FN={len(counts['fn'])}")

    run_gradcam_for_samples(
        model,
        test_dataset,
        sample_indices,
        thresholds,
        all_probs,
        all_preds,
        all_targets,
        device,
    )

    print(f"Grad-CAM figures saved under {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
