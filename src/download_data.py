import kagglehub
import shutil
from pathlib import Path

path = kagglehub.dataset_download("aryashah2k/soybean-seedsclassification-dataset")
print("Dataset descargado en:", path)

BASE_PATH = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_PATH / "data" / "raw"

if RAW_DATA_PATH.exists():
    shutil.rmtree(RAW_DATA_PATH)  
    
shutil.copytree(path, RAW_DATA_PATH)

print("Archivos copiados en:", RAW_DATA_PATH)
