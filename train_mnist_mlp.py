import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

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

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu", name="dense1", 
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(name="bn1"),
    tf.keras.layers.Dropout(0.3, name="dropout1"),
    tf.keras.layers.Dense(128, activation="relu", name="dense2",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(name="bn2"),
    tf.keras.layers.Dropout(0.3, name="dropout2"),
    tf.keras.layers.Dense(64, activation="relu", name="dense3",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(name="bn3"),
    tf.keras.layers.Dropout(0.2, name="dropout3"),
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
    x_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists("best_model.h5"):
    print("\nLoading best model weights...")
    model.load_weights("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\n{'='*50}")
print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"{'='*50}\n")

print("\nExtracting weights...")
dense1 = model.get_layer("dense1")
dense2 = model.get_layer("dense2")
dense3 = model.get_layer("dense3")
output = model.get_layer("output")

W1, b1 = dense1.get_weights()
W2, b2 = dense2.get_weights()
W3, b3 = dense3.get_weights()
W4, b4 = output.get_weights()

W1 = W1.T
W2 = W2.T
W3 = W3.T
W4 = W4.T

np.savez(
    "mnist_mlp_weights.npz",
    W1=W1, b1=b1,
    W2=W2, b2=b2,
    W3=W3, b3=b3,
    W4=W4, b4=b4,
)

print("✓ Saved weights to mnist_mlp_weights.npz")

if os.path.exists("best_model.h5"):
    os.remove("best_model.h5")
    print("✓ Cleaned up checkpoint file")
