import os
import datetime
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# Suppress TensorFlow warning
warnings.filterwarnings("ignore", category=UserWarning,
                        module='keras.src.trainers.data_adapters.py_dataset_adapter')

# Dataset paths
train_dir = r"C:\Users\Suganthan\Documents\kavin mini project\final model2\dataset\train"
valid_dir = r"C:\Users\Suganthan\Documents\kavin mini project\final model2\dataset\valid"

# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    zoom_range=0.4,
    shear_range=0.2,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1. / 255)

# Load training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Load validation data
val_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Print class labels
print("Class indices:", train_data.class_indices)

# Load MobileNetV2 base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False  # Freeze base model for initial training

# Functional model building
inputs = Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(6, activation='softmax')(x)  # 6 classes

model = Model(inputs, outputs)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TensorBoard logs
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Callbacks
callbacks = [
    EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('retinal_multiclass6_best_model.h5', save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# ====== INITIAL TRAINING ======
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    callbacks=callbacks
)

# Save model after initial training
model.save('retinal_multiclass6_initial_model.h5')
print("✅ Initial model saved as 'retinal_multiclass6_initial_model.h5'")

# ====== FINE-TUNING PHASE ======
# Unfreeze base model from a certain layer
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile with low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
fine_tune_epochs = 10
total_epochs = len(history.history['loss']) + fine_tune_epochs

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=callbacks
)

# Save final model
model.save('retinal_multiclass6_finetuned_model.h5')
print("✅ Fine-tuned model saved as 'retinal_multiclass6_finetuned_model.h5'")
