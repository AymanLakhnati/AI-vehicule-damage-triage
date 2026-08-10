import json
from pathlib import Path

path = Path('colab_setup.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "# Verify dataset paths and mount Google Drive if the dataset is missing\n",
    "from pathlib import Path\n",
    "import shutil\n",
    "import subprocess\n",
    "\n",
    "repo_root = Path('/content/vehicle-damage-triage')\n",
    "data_root = repo_root / 'data/raw/cardd/CarDD_release/CarDD_COCO'\n",
    "print('Data root exists:', data_root.exists())\n",
    "print('Train annotation exists:', (data_root / 'annotations/instances_train2017.json').exists())\n",
    "print('Val annotation exists:', (data_root / 'annotations/instances_val2017.json').exists())\n",
    "print('Train images dir exists:', (data_root / 'train2017').exists())\n",
    "print('Val images dir exists:', (data_root / 'val2017').exists())\n",
    "\n",
    "if not data_root.exists() or not (data_root / 'annotations/instances_train2017.json').exists():\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive', force_remount=True)\n",
    "    print('\\nMounted Drive. Searching for CarDD dataset in Drive and workspace...')\n",
    "    proc = subprocess.run(['find', '/content', '-name', 'instances_train2017.json'], capture_output=True, text=True)\n",
    "    found = proc.stdout.strip().splitlines()\n",
    "    if found:\n",
    "        print('Found instances_train2017.json at:')\n",
    "        for path in found:\n",
    "            print('  ' + path)\n",
    "        dataset_file = Path(found[0])\n",
    "        dataset_dir = dataset_file.parents[1]\n",
    "        print('\\nAssuming dataset root is:', dataset_dir)\n",
    "        print('Linking or copying dataset into the repo path:', data_root)\n",
    "        data_root.parent.mkdir(parents=True, exist_ok=True)\n",
    "        if data_root.exists():\n",
    "            print('Existing data_root already exists:', data_root)\n",
    "        else:\n",
    "            try:\n",
    "                data_root.symlink_to(dataset_dir, target_is_directory=True)\n",
    "                print('Created symlink:', data_root, '->', dataset_dir)\n",
    "            except Exception as exc:\n",
    "                print('Symlink failed, copying dataset instead:', exc)\n",
    "                shutil.copytree(dataset_dir, data_root)\n",
    "                print('Copied dataset to:', data_root)\n",
    "        print('\\nRechecking dataset path existence...')\n",
    "        print('Data root exists:', data_root.exists())\n",
    "        print('Train annotation exists:', (data_root / 'annotations/instances_train2017.json').exists())\n",
    "    else:\n",
    "        print('No CarDD dataset file found anywhere under /content.')\n",
    "        print('Please upload the dataset to Colab or mount it from Drive, then rerun this cell.')\n",
    "\n",
    "    proc2 = subprocess.run(['find', '/content', '-type', 'd', '-name', 'CarDD_COCO'], capture_output=True, text=True)\n",
    "    dirs = proc2.stdout.strip().splitlines()\n",
    "    if dirs:\n",
    "        print('\\nFound CarDD_COCO directories:')\n",
    "        for d in dirs:\n",
    "            print('  ' + d)\n",
    "        print('\\nIf one of these is the dataset, copy it into the repo path or update the training script paths.')\n",
]

# Replace cell at index 4
nb['cells'][4]['source'] = new_source

with path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print('Notebook patched successfully.')
