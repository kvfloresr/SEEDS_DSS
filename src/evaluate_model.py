from pathlib import Path
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"
model = tf.keras.models.load_model(MODEL_DIR / "best_model.keras")

IMG_SIZE = (128,128)
BATCH = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode='int',
    shuffle=False
    )

y_true = np.concatenate([y.numpy() for x,y in test_ds], axis=0)
x_all = np.concatenate([x.numpy() for x,y in test_ds], axis=0) 
y_prob = model.predict(x_all, batch_size=BATCH)
y_pred = np.argmax(y_prob, axis=1)

with open(MODEL_DIR / "class_indices.json","r",encoding="utf-8") as f:
    idx2class = json.load(f)
labels = [idx2class[str(i)] if str(i) in idx2class else idx2class[i] for i in range(len(idx2class))]

print("\n--- Classification report ---")
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(MODEL_DIR / "confusion_matrix.png", bbox_inches='tight')
print("Matriz de confusión guardada en models/confusion_matrix.png")
