import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Colab Training & Evaluation Setup\n",
                "Run this notebook on Google Colab to prepare the repository, install dependencies, and execute object detection training and evaluation.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required Python packages\n",
                "!pip install -q torch torchvision torchaudio torchmetrics pycocotools\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone the repository if needed and switch to it\n",
                "import os\n",
                "repo_dir = '/content/vehicle-damage-triage'\n",
                "if not os.path.exists(repo_dir):\n",
                "    !git clone https://github.com/AymanLakhnati/AI-vehicule-damage-triage.git {repo_dir}\n",
                "else:\n",
                "    print('Repository already exists at', repo_dir)\n",
                "os.chdir(repo_dir)\n",
                "print('Working directory:', os.getcwd())\n",
                "!ls -la\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Verify dataset paths and mount Google Drive if the dataset is missing\n",
                "from pathlib import Path\n",
                "data_root = Path('data/raw/cardd/CarDD_release/CarDD_COCO')\n",
                "print('Data root exists:', data_root.exists())\n",
                "print('Train annotation exists:', (data_root / 'annotations/instances_train2017.json').exists())\n",
                "print('Val annotation exists:', (data_root / 'annotations/instances_val2017.json').exists())\n",
                "print('Train images dir exists:', (data_root / 'train2017').exists())\n",
                "print('Val images dir exists:', (data_root / 'val2017').exists())\n",
                "if not data_root.exists():\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive', force_remount=True)\n",
                "    print('Mounted Drive. Update the dataset path if needed.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run training\n",
                "!python -u src/train_cardd_detector.py\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Confirm checkpoint files were created\n",
                "!ls -la models || true\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run evaluation\n",
                "!python src/evaluate_cardd_detector.py\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

Path('colab_setup.ipynb').write_text(json.dumps(nb, indent=2), encoding='utf-8')
