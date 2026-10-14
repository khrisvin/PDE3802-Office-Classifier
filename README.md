# PDE3802 Office Classifier

Group coursework for Artificial Intelligence in Robotics.

## Team
- Khrisvin Nepaulsing
- Veedoor Gopal
- Pranay Poheith

## Current progress
- Pilot dataset: 90 photographs across six classes.
- Classes: book, bottle, keyboard, mouse, mug and stapler.
- Image preview and dataset checking scripts implemented.
- Classification model not yet implemented.

## Setup
Install uv, then run these commands from the project folder:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Run the image preview

```powershell
& ".\.venv\Scripts\python.exe" check_image.py
```

Click the image window and press any key to close it.

## Check the dataset

```powershell
& ".\.venv\Scripts\python.exe" check_dataset.py
```

Expected output for the current pilot dataset:
- 15 photographs per class
- 90 total
- 90 readable
- 0 unreadable

The checker currently counts and reads files ending in .jpeg.

## Troubleshooting
- Missing script: run commands from the project folder.
- Missing image: check the image filename and its location under data/raw.
- Missing library: repeat the dependency installation command above.