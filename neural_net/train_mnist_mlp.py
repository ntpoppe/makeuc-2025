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

def load_custom_eights(folder_path: str):
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if not image_paths:
        return None, None

    images = []
    labels = []
    for image_path in image_paths:
        try:
            img = keras.utils.load_img(
                image_path,
                color_mode="grayscale",
                target_size=(28, 28),
            )
            arr = keras.utils.img_to_array(img).astype("float32") / 255.0
            arr = np.squeeze(arr, axis=-1)
            images.append(arr)
            labels.append(8)
        except Exception as exc:
            print(f"Skipping {image_path}: {exc}")

    if not images:
        return None, None

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype="int64")
    return images, labels


def load_custom_twos(folder_path: str):
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if not image_paths:
        return None, None

    images = []
    labels = []
    for image_path in image_paths:
        try:
            img = keras.utils.load_img(
                image_path,
                color_mode="grayscale",
                target_size=(28, 28),
            )
            arr = keras.utils.img_to_array(img).astype("float32") / 255.0
            arr = np.squeeze(arr, axis=-1)
            images.append(arr)
            labels.append(2)
        except Exception as exc:
            print(f"Skipping {image_path}: {exc}")

    if not images:
        return None, None

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype="int64")
    return images, labels


def build_custom_eight_augmentation() -> tf.keras.Sequential:
    """Create a strong augmentation pipeline for handwritten '8's."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.12,
                width_factor=0.12,
                fill_mode="reflect",
                name="aug_translate",
            ),
            tf.keras.layers.RandomRotation(
                factor=0.2,
                fill_mode="reflect",
                name="aug_rotate",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.25, 0.18),
                width_factor=(-0.25, 0.18),
                fill_mode="reflect",
                name="aug_zoom",
            ),
            tf.keras.layers.RandomContrast(
                factor=0.25,
                name="aug_contrast",
            ),
            tf.keras.layers.RandomBrightness(
                factor=0.2,
                name="aug_brightness",
            ),
            tf.keras.layers.GaussianNoise(
                stddev=0.04,
                name="aug_noise",
            ),
            tf.keras.layers.RandomCrop(
                height=26,
                width=26,
                name="aug_crop",
            ),
            tf.keras.layers.Resizing(
                height=28,
                width=28,
                interpolation="bilinear",
                name="aug_resize",
            ),
        ],
        name="custom_eight_augmentation",
    )


def split_custom_eights(
    images: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split handwritten '8's into train/val/test portions."""
    total = len(images)
    if total == 0:
        empty = np.empty((0, 28, 28), dtype="float32")
        empty_labels = np.empty((0,), dtype="int64")
        return empty, empty_labels, empty, empty_labels, empty, empty_labels

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    images = images[indices]
    labels = labels[indices]

    val_count = max(1, int(round(total * val_fraction))) if total >= 5 else max(0, total - 2)
    test_count = max(1, int(round(total * test_fraction))) if total >= 5 else max(0, total - val_count - 1)
    train_count = total - val_count - test_count

    train_x = images[:train_count]
    train_y = labels[:train_count]
    val_x = images[train_count:train_count + val_count]
    val_y = labels[train_count:train_count + val_count]
    test_x = images[train_count + val_count:]
    test_y = labels[train_count + val_count:]

    return train_x, train_y, val_x, val_y, test_x, test_y


custom_eight_dir = os.path.join(os.path.dirname(__file__), "8s")
custom_eight_x, custom_eight_y = (None, None)
custom_eight_train_x = custom_eight_train_y = None
if os.path.isdir(custom_eight_dir):
    custom_eight_x, custom_eight_y = load_custom_eights(custom_eight_dir)
    if custom_eight_x is not None:
        (
            custom_eight_train_x,
            custom_eight_train_y,
            custom_eight_val_x,
            custom_eight_val_y,
            custom_eight_test_x,
            custom_eight_test_y,
        ) = split_custom_eights(custom_eight_x, custom_eight_y)

        if len(custom_eight_val_x):
            x_val = np.concatenate([x_val, custom_eight_val_x], axis=0)
            y_val = np.concatenate([y_val, custom_eight_val_y], axis=0)
        if len(custom_eight_test_x):
            x_test = np.concatenate([x_test, custom_eight_test_x], axis=0)
            y_test = np.concatenate([y_test, custom_eight_test_y], axis=0)

        x_train = np.concatenate([x_train, custom_eight_train_x], axis=0)
        y_train = np.concatenate([y_train, custom_eight_train_y], axis=0)
        print(
            f"Integrated {len(custom_eight_train_x)} custom '8' samples into training, "
            f"{len(custom_eight_val_x)} into validation, and {len(custom_eight_test_x)} into test from {custom_eight_dir}"
        )

custom_two_dir = os.path.join(os.path.dirname(__file__), "2s")
custom_two_x, custom_two_y = (None, None)
custom_two_train_x = custom_two_train_y = None
if os.path.isdir(custom_two_dir):
    custom_two_x, custom_two_y = load_custom_twos(custom_two_dir)
    if custom_two_x is not None:
        (
            custom_two_train_x,
            custom_two_train_y,
            custom_two_val_x,
            custom_two_val_y,
            custom_two_test_x,
            custom_two_test_y,
        ) = split_custom_eights(custom_two_x, custom_two_y)

        if len(custom_two_val_x):
            x_val = np.concatenate([x_val, custom_two_val_x], axis=0)
            y_val = np.concatenate([y_val, custom_two_val_y], axis=0)
        if len(custom_two_test_x):
            x_test = np.concatenate([x_test, custom_two_test_x], axis=0)
            y_test = np.concatenate([y_test, custom_two_test_y], axis=0)

        x_train = np.concatenate([x_train, custom_two_train_x], axis=0)
        y_train = np.concatenate([y_train, custom_two_train_y], axis=0)
        print(
            f"Integrated {len(custom_two_train_x)} custom '2' samples into training, "
            f"{len(custom_two_val_x)} into validation, and {len(custom_two_test_x)} into test from {custom_two_dir}"
        )

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")

batch_size = 128

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(len(x_train), reshuffle_each_iteration=True)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

if custom_eight_train_x is not None and len(custom_eight_train_x):
    augmentation_layer = build_custom_eight_augmentation()

    def augment_custom(image: tf.Tensor, label: tf.Tensor):
        image = tf.expand_dims(image, -1)
        image = augmentation_layer(image, training=True)
        image = tf.squeeze(image, axis=-1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    custom_aug_ds = tf.data.Dataset.from_tensor_slices((custom_eight_train_x, custom_eight_train_y))
    custom_aug_ds = custom_aug_ds.shuffle(len(custom_eight_train_x), reshuffle_each_iteration=True)
    custom_aug_ds = custom_aug_ds.map(augment_custom, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.concatenate(custom_aug_ds)

total_train_samples = len(x_train)
if custom_eight_train_x is not None and len(custom_eight_train_x):
    total_train_samples += len(custom_eight_train_x)

steps_per_epoch = math.ceil(total_train_samples / batch_size)
train_ds = train_ds.shuffle(total_train_samples, reshuffle_each_iteration=True)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
