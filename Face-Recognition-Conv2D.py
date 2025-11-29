import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import os
import matplotlib.pyplot as plt

import optuna
import wandb 
import gc

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("TensorFlow is using the GPU \n", gpus)
else:
    print("No GPU detected.")
    
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

gc.collect()
tf.keras.backend.clear_session()

from wandb.integration.keras import WandbMetricsLogger

wandb.require("core")
wandb.login()

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, (160, 160))
    return img, label

def make_dataset(paths, labels, batch_size = 32, shuffle = False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
ds = pd.read_csv("/tf/Face-Recognition/CelebA/identity_CelebA.txt", sep = r"\s+", names=["image", "identity"])

ds["identity"] = 0
ds.head()

ds = ds.sample(200, random_state = 4).reset_index(drop = True)
ds.head()

image = np.array([])
identity = np.zeros(136)

for i in range(136):
    image = np.append(image, f"me{i+1}.jpg")
    identity[i] = 1

identity = identity.astype(int)

df = pd.DataFrame({
    "image": image,
    "identity": identity,
})

ds = pd.concat([ds, df], axis = 0)
ds = ds.sample(frac = 1, random_state = 5).reset_index(drop = True)

df_train, df_temp = train_test_split(
    ds, test_size = 0.4, stratify = ds["identity"], random_state = 5)

df_val, df_test = train_test_split(
    df_temp, test_size = 0.5, stratify = df_temp["identity"], random_state = 5)

print(df_train["identity"].value_counts())
print(df_val["identity"].value_counts())
print(df_test["identity"].value_counts())

def reset_dataset_dir(base="data"):
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(os.path.join(base, "train/me"))
    os.makedirs(os.path.join(base, "train/others"))
    os.makedirs(os.path.join(base, "val/me"))
    os.makedirs(os.path.join(base, "val/others"))
    os.makedirs(os.path.join(base, "test/me"))
    os.makedirs(os.path.join(base, "test/others"))

import shutil

base = "faces"

for split in ["train", "val", "test"]:
    os.makedirs(f"{base}/{split}/me", exist_ok = True)
    os.makedirs(f"{base}/{split}/others", exist_ok = True)

def copy_files(df, split):
    for idx, row in df.iterrows():
        if row["identity"] == 1:
            src = os.path.join("/tf/Face-Recognition/CelebA/Me", row["image"])
            dst = f"{base}/{split}/me/{row['image']}"
        else:
            src = os.path.join("/tf/Face-Recognition/CelebA/img_align_celeba", row["image"])
            dst = f"{base}/{split}/others/{row['image']}"
        
        shutil.copy(src, dst)
        
reset_dataset_dir()

copy_files(df_train, "train")
copy_files(df_val, "val")
copy_files(df_test, "test")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True)

test_val_datagen = ImageDataGenerator()

train = train_datagen.flow_from_directory(
    f"{base}/train",
    target_size = (160,160),
    class_mode = "binary",
    batch_size = 32,
    seed = 4,
    shuffle = True)

val = test_val_datagen.flow_from_directory(
    f"{base}/val",
    target_size = (160,160),
    class_mode = "binary",
    batch_size = 32,
    seed = 4,
    shuffle = False)

test = test_val_datagen.flow_from_directory(
    f"{base}/test",
    target_size = (160,160),
    class_mode = "binary",
    batch_size = 32,
    seed = 4,
    shuffle = False)

FeatureExtractor = tf.keras.models.load_model("/tf/Face-Recognition/Models/Conv2D-MobileNetV2-Based-Fine-Tunned.keras")
FeatureExtractor.trainable = False

FeatureExtractor_Output = FeatureExtractor.layers[-3].output
FeatureExtractor_Model = tf.keras.Model(FeatureExtractor.input, FeatureExtractor_Output)

for layer in FeatureExtractor_Model.layers:
    layer.trainable = False

lr = 1e-4
optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr) 

inputs = tf.keras.Input(shape=(160,160,3))
x = tf.keras.layers.Dense(256, activation="leaky_relu")(FeatureExtractor_Model.output)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation = "leaky_relu")(x)
x = tf.keras.layers.Dropout(0.15)(x)
x = tf.keras.layers.Dense(256, activation = "relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)

outputs = tf.keras.layers.Dense(1, activation = "sigmoid", dtype = "float32",name = "classifier_head")(x)

model = tf.keras.Model(inputs = FeatureExtractor_Model.input, outputs = outputs)

model.compile(loss = "binary_crossentropy",
              optimizer = optimizer,
              metrics = ["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor = "val_accuracy", patience = 10, restore_best_weights = True)
lr_reduction = ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 7)

wandb.init(
        project = "Face-Recognition-Conv2D-Trials-Exp-Series1.0",
        name = "Trial_1_FullSet",
        reinit = True,
        config = {
            "activation": "leaky_relu, relu",
            "n_layers": 3,
            "learning_rate": lr,
            "optimizer": "RMSProp"
        }
    )

history = model.fit(
    train, 
    validation_data = val,
    epochs = 200,
    verbose = 1, 
    callbacks = [WandbMetricsLogger(log_freq = 5), early_stopping, lr_reduction]
        )

model.save("Face-Recognition-Conv2D.keras")

tf.keras.backend.clear_session()
wandb.finish()
gc.collect()

#  Fine-Tunning

for layer in FeatureExtractor_Model.layers[:-5]:
    layer.trainable = False

for layer in FeatureExtractor_Model.layers[-5:]:
    layer.trainable = True

optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-6)

model.compile(loss = "binary_crossentropy",
              optimizer = optimizer,
              metrics = ["accuracy"])

wandb.init(
        project = "Conv2D-MobileNetV2-Based-Trials-Exp-Series1.0",
        name = "FineTunning_1_FullSet",
        reinit = True,
        config = {
            "activation": "leaky_relu, relu",
            "n_layers": 3,
            "learning_rate": lr,
            "optimizer": "RMSProp"
        }
    )

history = model.fit(
    train, 
    validation_data = val,
    epochs = 10,
    verbose = 1)

model.save("Face-Recognition-Conv2D-Fine-Tunned.keras") 

tf.keras.backend.clear_session()
wandb.finish()
gc.collect()

model_ev = keras.models.load_model("Face-Recognition-Conv2D-Fine-Tunned.keras")

loss, accuracy = model_ev.evaluate(test, verbose = 1)

print(f"Test Loss: {loss:}")
print(f"Test Accuracy: {accuracy:}")

from sklearn.metrics import confusion_matrix

pred_probs = model.predict(test)

y_pred = (pred_probs > 0.5).astype("int32")
y_pred = y_pred.reshape(-1)

y_true = test.classes

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot = True, fmt = "d", cmap = "rocket")

plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.show()