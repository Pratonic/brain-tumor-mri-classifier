"""
Brain Tumor MRI Classifier — Inference Script
Loads the trained model and runs prediction on any image you give it.

Usage:
    python predict.py --image "path/to/mri_scan.jpg"
    python predict.py --image "path/to/mri_scan.jpg" --gradcam
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

IMG_SIZE   = 260
MODEL_PATH = "outputs/brain_tumor_model.keras"

# Class names are saved alongside the model so this script works standalone
# If you have more/fewer classes, this gets auto-read from the saved file
CLASSES_FILE = "outputs/class_names.txt"


def load_class_names():
    if not os.path.exists(CLASSES_FILE):
        raise FileNotFoundError(
            f"Class names file not found at '{CLASSES_FILE}'. "
            "Run train.py first — it saves this file automatically."
        )
    with open(CLASSES_FILE) as f:
        return [line.strip() for line in f.readlines()]


def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)


def make_gradcam_heatmap(img_tensor, model):
    try:
        base      = model.get_layer('efficientnetb2')
        last_conv = base.get_layer('top_conv')
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output])

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_out, preds = grad_model(img_tensor)
            pred_idx      = tf.argmax(preds[0])
            class_channel = preds[:, pred_idx]

        grads        = tape.gradient(class_channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"  Grad-CAM failed: {e}")
        return None


def overlay_gradcam(img_path, heatmap, alpha=0.4):
    import cv2
    img             = cv2.imread(img_path)
    img             = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colormap        = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed    = cv2.addWeighted(img, 1 - alpha, colormap, alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


def predict(img_path, model, class_names, top_k=5, show_gradcam=False):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_tensor = preprocess_image(img_path)
    probs      = model.predict(img_tensor, verbose=0)[0]
    top_idx    = np.argsort(probs)[::-1][:top_k]

    print(f"\nImage: {img_path}")
    print(f"Top-{top_k} predictions:")
    print("-" * 45)
    for i, idx in enumerate(top_idx):
        bar = "█" * int(probs[idx] * 30)
        print(f"  {i+1}. {class_names[idx]:<30} {probs[idx]:.2%}  {bar}")

    # Plot
    cols  = 2 if show_gradcam else 1
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1:
        axes = [axes]

    orig = np.array(Image.open(img_path).convert('RGB'))
    axes[0].imshow(orig)
    axes[0].set_title(
        f"Prediction: {class_names[top_idx[0]]}\n({probs[top_idx[0]]:.1%} confidence)",
        fontsize=10)
    axes[0].axis('off')

    if show_gradcam:
        heatmap = make_gradcam_heatmap(img_tensor, model)
        if heatmap is not None:
            overlay = overlay_gradcam(img_path, heatmap)
            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM attention", fontsize=10)
        else:
            axes[1].imshow(orig)
            axes[1].set_title("Grad-CAM unavailable", fontsize=10)
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',   required=True, help='Path to MRI image')
    parser.add_argument('--gradcam', action='store_true',
                        help='Show Grad-CAM heatmap alongside prediction')
    parser.add_argument('--topk',    type=int, default=5,
                        help='How many top predictions to show (default: 5)')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at '{MODEL_PATH}'.")
        print("Run train.py first to train and save the model.")
        return

    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    print(f"Loaded. {len(class_names)} classes.\n")

    predict(args.image, model, class_names,
            top_k=args.topk, show_gradcam=args.gradcam)


if __name__ == '__main__':
    main()
