# Upload to GitHub

Create a new GitHub repository (for example `AE5510-YOLOv8`) and then run:

```bash
git clone https://github.com/<your-username>/AE5510-YOLOv8.git
cd AE5510-YOLOv8
```

Copy the contents of this folder into the cloned repository, then:

```bash
git add .
git commit -m "Add YOLOv8 UAS object detection baseline"
git push origin main
```

Do **not** add the original aerial-image dataset or the generated `yolo_dataset/`; the `.gitignore` already excludes them.

The repository includes the trained `best.pt` and `last.pt` checkpoints, training artifacts, source scripts, and reproducibility documentation.
