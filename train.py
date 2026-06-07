"""
Brain Tumor MRI Classifier — Training Script
Run once to train and save the model. Subsequent runs skip training.

Usage:
    python train.py --data "C:/path/to/dataset"
    python train.py --data "/home/you/dataset"
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = 260
BATCH_SIZE  = 32        # drop to 16 if you get OOM
EPOCHS_HEAD = 10
EPOCHS_FINE = 15
LR_HEAD     = 1e-3
LR_FINE     = 1e-4
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
SEED        = 42
MODEL_PATH  = "outputs/brain_tumor_model.keras"   # trained model lives here
OUTPUT_DIR  = "outputs"
AUTOTUNE    = tf.data.AUTOTUNE


# ── Helpers ──────────────────────────────────────────────────────────────────
def find_data_root(base):
    for root, dirs, _ in os.walk(base):
        subdirs = [d for d in dirs if not d.startswith('.')]
        if len(subdirs) >= 5:
            first = os.path.join(root, subdirs[0])
            imgs  = [f for f in os.listdir(first)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                return root
    return base


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.rot90(img, k=tf.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_model(num_classes):
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB2

    base = EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs), base


def plot_history(h1, h2):
    acc   = h1.history['accuracy']     + h2.history['accuracy']
    val   = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss  = h1.history['loss']         + h2.history['loss']
    vloss = h1.history['val_loss']     + h2.history['val_loss']
    ep    = range(1, len(acc) + 1)
    split = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for ax in (ax1, ax2):
        ax.axvline(split + 0.5, color='gray', linestyle='--',
                   alpha=0.5, label='Fine-tune start')
    ax1.plot(ep, acc, label='Train',   color='royalblue')
    ax1.plot(ep, val, label='Val',     color='tomato')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.set_xlabel('Epoch')
    ax2.plot(ep, loss,  label='Train', color='royalblue')
    ax2.plot(ep, vloss, label='Val',   color='tomato')
    ax2.set_title('Loss'); ax2.legend(); ax2.set_xlabel('Epoch')
    plt.suptitle('Training History', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"  Saved training curves → {OUTPUT_DIR}/training_curves.png")


def save_eval_plots(model, test_ds, X_test, y_test, class_names):
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = np.array(y_test)

    # Confusion matrix
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Normalised Confusion Matrix', fontweight='bold')
    plt.xticks(rotation=75, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {OUTPUT_DIR}/confusion_matrix.png")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print(f"  Saved classification report → {OUTPUT_DIR}/classification_report.txt")

    return y_pred_probs, y_pred, y_true


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True,
                        help='Path to your dataset folder (with class subfolders)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if a saved model exists')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {[g.name for g in gpus]}")
    else:
        print("No GPU detected — running on CPU (will be slow)")

    # Dataset structure
    data_dir    = find_data_root(args.data)
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    num_classes = len(class_names)
    print(f"\nDataset : {data_dir}")
    print(f"Classes : {num_classes} → {class_names[:5]}{'...' if num_classes > 5 else ''}")

    # Collect all file paths
    all_paths, all_labels = [], []
    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(data_dir, cls)
        for f in os.listdir(cls_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(cls_path, f))
                all_labels.append(idx)
    print(f"Total images: {len(all_paths)}")

    # Train/val/test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        all_paths, all_labels, test_size=TEST_SPLIT,
        stratify=all_labels, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=y_tv, random_state=SEED)
    print(f"Split → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds   = make_dataset(X_val,   y_val)
    test_ds  = make_dataset(X_test,  y_test)

    # ── Load or Train ─────────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH) and not args.retrain:
        print(f"✅ Saved model found at '{MODEL_PATH}' — loading, skipping training.")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        if args.retrain:
            print("--retrain flag set, retraining from scratch.\n")
        else:
            print("No saved model found, starting training.\n")

        # Mixed precision (comment out if you get issues on older GPUs)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        model, base_model = build_model(num_classes)

        class_weights_arr = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train)
        class_weight_dict = dict(enumerate(class_weights_arr))

        # Phase 1 — head only
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR_HEAD),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        print("=== PHASE 1: Training head ===")
        history1 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS_HEAD,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=4, restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=2, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(OUTPUT_DIR, 'checkpoint_phase1.keras'),
                    save_best_only=True, verbose=0),
            ])

        # Phase 2 — fine-tune last 30 backbone layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR_FINE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        print("\n=== PHASE 2: Fine-tuning ===")
        history2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS_FINE,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=3, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(OUTPUT_DIR, 'checkpoint_phase2.keras'),
                    save_best_only=True, verbose=0),
            ])

        # Save final model
        model.save(MODEL_PATH)
        print(f"\n✅ Model saved → {MODEL_PATH}")

        # Save class names so predict.py can work standalone
        with open(os.path.join(OUTPUT_DIR, 'class_names.txt'), 'w') as f:
            f.write('\n'.join(class_names))
        print(f"   Class names saved → {OUTPUT_DIR}/class_names.txt")

        plot_history(history1, history2)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n=== Evaluating on test set ===")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    y_pred_probs = model.predict(test_ds, verbose=0)
    top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    top5.update_state(np.array(y_test), y_pred_probs)
    print(f"Top-5 Accuracy: {top5.result().numpy():.4f}")

    save_eval_plots(model, test_ds, X_test, y_test, class_names)
    print("\nDone. All outputs saved to ./outputs/")


if __name__ == '__main__':
    main()
