import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_PATH = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_PATH / "data" / "raw"
PROCESSED_DATA_PATH = BASE_PATH / "data" / "processed"

for split in ["train", "val", "test"]:
    (PROCESSED_DATA_PATH / split).mkdir(parents=True, exist_ok=True)

classes = [d.name for d in RAW_DATA_PATH.iterdir() if d.is_dir()]
print(f"Clases detectadas: {classes}")

def list_images(folder: Path, exts=(".bmp", ".jpg", ".jpeg", ".png")):
    files = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
    return files

for cls in classes:
    cls_files = list_images(RAW_DATA_PATH / cls)
    if len(cls_files) == 0:
        print(f"No se encontraron imágenes para la clase {cls}")
        continue

    train_files, temp_files = train_test_split(cls_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    def copy_files(file_list, split):
        dest_dir = PROCESSED_DATA_PATH / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in file_list:
            shutil.copy(f, dest_dir / f.name)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"{cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("Preprocesamiento completado.")
