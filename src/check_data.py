from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_PATH / "data" / "raw"
PROCESSED_DATA_PATH = BASE_PATH / "data" / "processed"

def count_images_in_directory(directory: Path):
    """Cuenta las imágenes .bmp en las subcarpetas de clases."""
    summary = {}
    if not directory.exists():
        print(f"El directorio {directory} no existe.")
        return summary
    
    for cls_dir in directory.iterdir():
        if cls_dir.is_dir():
            num_images = len(list(cls_dir.glob("*.bmp","*.jpg","*.jpeg","*.png")))
            summary[cls_dir.name] = num_images
    return summary

def print_summary():
    print("\n Resumen de datos:")

    print("\n--- DATASET ORIGINAL (RAW) ---")
    raw_summary = count_images_in_directory(RAW_DATA_PATH)
    for cls, count in raw_summary.items():
        print(f"{cls}: {count} imágenes")

    print("\n--- DATASET PROCESADO (train/val/test) ---")
    for split in ["train", "val", "test"]:
        split_dir = PROCESSED_DATA_PATH / split
        print(f"\n🔹 {split.upper()}:")
        split_summary = count_images_in_directory(split_dir)
        for cls, count in split_summary.items():
            print(f"   {cls}: {count} imágenes")

if __name__ == "__main__":
    print_summary()
