import os
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "data/processed"

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Clases detectadas:", class_names)

# --- Pesos por clase calculados automaticamente ---
# Cuenta cuantas imagenes hay en cada carpeta de clase y arma los pesos.
# Se adapta solo a cualquier numero de clases (5, 6, etc.), asi que nunca
# vuelve a fallar por un desajuste como el del diccionario anterior.
train_dir = os.path.join(DATA_DIR, "train")
counts = []
for cname in class_names:
    cdir = os.path.join(train_dir, cname)
    n = len([f for f in os.listdir(cdir)
            if os.path.isfile(os.path.join(cdir, f))])
    counts.append(n)

total = sum(counts)
class_weights_dict = {
    i: total / (num_classes * c) for i, c in enumerate(counts)
}
print("Cantidad por clase:", dict(zip(class_names, counts)))
print("Pesos por clase:", class_weights_dict)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Rescaling(1./255),
    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_uniform'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint("models/best_model.keras", save_best_only=True) 
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

MODEL_PATH = "models/seed_cnn.keras"
model.save(MODEL_PATH)


import json
with open("models/class_indices.json", "w", encoding="utf-8") as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f, indent=4)