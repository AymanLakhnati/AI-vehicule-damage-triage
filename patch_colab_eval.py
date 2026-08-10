import json
from pathlib import Path

path = Path('colab_setup.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

new_eval = [
    "# Run evaluation only if a detector checkpoint exists\n",
    "from pathlib import Path\n",
    "checkpoint_dir = Path('models')\n",
    "checkpoints = sorted(checkpoint_dir.glob('cardd_detector_epoch*.pth'))\n",
    "print('Detected checkpoint files:')\n",
    "for ckpt in checkpoints:\n",
    "    print(' ', ckpt)\n",
    "\n",
    "if checkpoints:\n",
    "    import subprocess\n",
    "    subprocess.run(['python', 'src/evaluate_cardd_detector.py'])\n",
    "else:\n",
    "    print('No detector checkpoint found in models/. Train first or copy a checkpoint into this directory.')\n",
]
nb['cells'][7]['source'] = new_eval
with path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print('Evaluation cell patched successfully.')
