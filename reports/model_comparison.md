# Binary Classification Experiment — Model Comparison

| Model | Accuracy | Class 1 Recall | Class 1 F1 | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| CNN — Unweighted | 0.9422 | 0.0000 | 0.0000 | 0.4851 | 0.5000 |
| CNN — Weighted | 0.8300 | 0.5286 | 0.2643 | 0.5841 | 0.6885 |
| ResNet18 — Frozen + Weighted | 0.8465 | 0.8857 | 0.4000 | 0.6560 | 0.8649 |
| ResNet18 — Fine-tuned + Weighted | **0.9629** | **0.9000** | **0.7368** | **0.8584** | **0.9334** |

## Key Findings

1. Accuracy alone was misleading because the dataset was highly imbalanced.
2. The unweighted CNN achieved 94.22% accuracy by predicting only the majority class.
3. Class-weighted loss significantly improved minority-class detection.
4. Transfer learning with ResNet18 substantially outperformed the CNN trained from scratch.
5. Fine-tuning the final ResNet block produced the strongest model.
6. The final fine-tuned model detected 63 of the 70 class-1 test examples.
7. This dataset is retained as an auxiliary binary-classification experiment and is not used as the main vehicle-damage-type dataset.
