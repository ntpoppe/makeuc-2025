import glob
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

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

# Basic MLP: 784 → 32 → 10
# Single hidden layer with 32 neurons
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="leaky_relu", name="hidden1",
                          kernel_initializer="he_normal"),
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
hidden1 = model.get_layer("hidden1")
output = model.get_layer("output")

W1, b1 = hidden1.get_weights()
W2, b2 = output.get_weights()

# Transpose to match NumPy format (out_features, in_features)
W1 = W1.T
W2 = W2.T

np.savez(
    "mnist_mlp_weights.npz",
    W1=W1, b1=b1,
    W2=W2, b2=b2,
)

print("✓ Saved weights to mnist_mlp_weights.npz")

if os.path.exists("best_model.h5"):
    os.remove("best_model.h5")
    print("✓ Cleaned up checkpoint file")
