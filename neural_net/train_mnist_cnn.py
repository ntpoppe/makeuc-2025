import glob
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

val_size = int(0.1 * len(x_train))
x_val = x_train[:val_size]
y_val = y_train[:val_size]
x_train = x_train[val_size:]
y_train = y_train[val_size:]

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")

batch_size = 128

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(len(x_train), reshuffle_each_iteration=True)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

total_train_samples = len(x_train)
steps_per_epoch = math.ceil(total_train_samples / batch_size)
train_ds = train_ds.shuffle(total_train_samples, reshuffle_each_iteration=True)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    # Single convolutional block
    tf.keras.layers.Conv2D(4, (3, 3), activation="relu", name="conv1",
                           kernel_initializer="he_normal", padding="valid"),
    tf.keras.layers.BatchNormalization(name="bn_conv1"),
    tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
    # Flatten for dense layers
    tf.keras.layers.Flatten(name="flatten"),
    # Dense layers: 64, 64, 10
    tf.keras.layers.Dense(64, activation="relu", name="dense1", 
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(name="bn1"),
    tf.keras.layers.Dropout(0.3, name="dropout1"),
    tf.keras.layers.Dense(64, activation="relu", name="dense2",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(name="bn2"),
    tf.keras.layers.Dropout(0.3, name="dropout2"),
    tf.keras.layers.Dense(10, activation="softmax", name="output"),
])

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

history = model.fit(
    train_ds,
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists("best_model.h5"):
    print("\nLoading best model weights...")
    model.load_weights("best_model.h5")

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n{'='*50}")
print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"{'='*50}\n")

print("\nExtracting weights...")
conv1 = model.get_layer("conv1")
bn_conv1 = model.get_layer("bn_conv1")
dense1 = model.get_layer("dense1")
bn1 = model.get_layer("bn1")
dense2 = model.get_layer("dense2")
bn2 = model.get_layer("bn2")
output = model.get_layer("output")

# Extract conv layer weights (filters are in shape: (kernel_h, kernel_w, in_channels, out_channels))
conv1_kernel, conv1_bias = conv1.get_weights()

# Extract batch norm parameters
bn_conv1_gamma, bn_conv1_beta, bn_conv1_mean, bn_conv1_var = bn_conv1.get_weights()

# Extract dense layer weights
dense1_kernel, dense1_bias = dense1.get_weights()
dense2_kernel, dense2_bias = dense2.get_weights()
output_kernel, output_bias = output.get_weights()

# Extract batch norm for dense layers
bn1_gamma, bn1_beta, bn1_mean, bn1_var = bn1.get_weights()
bn2_gamma, bn2_beta, bn2_mean, bn2_var = bn2.get_weights()

# Transpose dense weights to match NumPy format (out_features, in_features)
dense1_kernel = dense1_kernel.T
dense2_kernel = dense2_kernel.T
output_kernel = output_kernel.T

np.savez(
    "mnist_cnn_weights.npz",
    # Conv layer 1
    conv1_kernel=conv1_kernel, conv1_bias=conv1_bias,
    bn_conv1_gamma=bn_conv1_gamma, bn_conv1_beta=bn_conv1_beta,
    bn_conv1_mean=bn_conv1_mean, bn_conv1_var=bn_conv1_var,
    # Dense layer 1
    dense1_kernel=dense1_kernel, dense1_bias=dense1_bias,
    bn1_gamma=bn1_gamma, bn1_beta=bn1_beta,
    bn1_mean=bn1_mean, bn1_var=bn1_var,
    # Dense layer 2
    dense2_kernel=dense2_kernel, dense2_bias=dense2_bias,
    bn2_gamma=bn2_gamma, bn2_beta=bn2_beta,
    bn2_mean=bn2_mean, bn2_var=bn2_var,
    # Output layer
    output_kernel=output_kernel, output_bias=output_bias,
)

print("✓ Saved weights to mnist_cnn_weights.npz")

if os.path.exists("best_model.h5"):
    os.remove("best_model.h5")
    print("✓ Cleaned up checkpoint file")
