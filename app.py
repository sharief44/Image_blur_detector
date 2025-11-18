# streamlit_blur_detector.py
# Complete Streamlit app: Laplacian blur detector + heatmap + threshold slider

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(page_title="AI Image Blur Detector — Laplacian", layout="wide")

st.title("🔎 AI Image Blur Detector — Laplacian Variance")
st.write("Upload an image, adjust the threshold, and inspect the blur heatmap and overlay.")

# Sidebar controls
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Laplacian variance threshold", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
show_overlay = st.sidebar.checkbox("Show heatmap overlay on original", value=True)
colormap_options = ["JET", "HOT", "VIRIDIS", "PLASMA"]
cmap_choice = st.sidebar.selectbox("Heatmap colormap", colormap_options, index=0)
resize_option = st.sidebar.checkbox("Resize large images to 1024px max (keeps speed)", value=True)

# Utility functions
COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "PLASMA": cv2.COLORMAP_PLASMA,
}


def read_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def to_rgb(cv_img):
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def laplacian_heatmap(gray):
    # Compute absolute Laplacian and normalize to 0-255
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.absolute(lap)
    # Normalize
    if lap_abs.max() == 0:
        norm = np.zeros_like(lap_abs, dtype=np.uint8)
    else:
        norm = np.uint8(255.0 * (lap_abs / lap_abs.max()))
    return norm


def apply_colormap(norm, cmap_name="JET"):
    cmap = COLORMAPS.get(cmap_name, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(norm, cmap)
    return colored


def overlay_heatmap_on_image(orig, heatmap, alpha=0.5):
    # orig, heatmap are BGR images
    # resize heatmap if necessary
    if orig.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    overlay = cv2.addWeighted(heatmap, alpha, orig, 1 - alpha, 0)
    return overlay

# Layout: two columns
uploaded = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded is not None:
    # Read image
    img = read_image(uploaded)
    if img is None:
        st.error("Couldn't read the uploaded file as an image.")
    else:
        # Optional resize to keep UI performant
        if resize_option:
            max_dim = max(img.shape[0], img.shape[1])
            if max_dim > 1024:
                scale = 1024.0 / max_dim
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        img_rgb = to_rgb(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute score and heatmap
        score = variance_of_laplacian(gray)
        heat_norm = laplacian_heatmap(gray)
        heat_colored = apply_colormap(heat_norm, cmap_choice)

        # Create overlay
        overlay_bgr = overlay_heatmap_on_image(img, heat_colored, alpha=0.55) if show_overlay else None
        overlay_rgb = to_rgb(overlay_bgr) if overlay_bgr is not None else None

        # Classification
        is_blurry = score < threshold

        # Display
        with col1:
            st.subheader("Original Image")
            st.image(img_rgb, use_column_width=True)
            st.markdown(f"**Laplacian variance score:** `{score:.2f}`")
            st.markdown(f"**Threshold:** `{threshold:.1f}` → **Blurry**: `{is_blurry}`")
            if is_blurry:
                st.warning("Image flagged as **blurry**. Increase threshold to be more permissive.")
            else:
                st.success("Image appears **sharp** (above threshold).")

        with col2:
            st.subheader("Blur Heatmap")
            st.image(to_rgb(heat_colored), use_column_width=True)
            st.caption("Heatmap shows per-pixel Laplacian magnitude (high = sharp edges).")

            if show_overlay and overlay_rgb is not None:
                st.subheader("Overlay (heatmap + original)")
                st.image(overlay_rgb, use_column_width=True)

        st.markdown("---")
        st.subheader("Advanced settings & diagnostics")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.write("**Image shape:**", img.shape)
            st.write("**Image dtype:**", img.dtype)
            st.write("**Max Laplacian value (abs):**", float(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).max()))
        with stats_col2:
            st.write("**Mean Laplacian (abs):**", float(np.abs(cv2.Laplacian(gray, cv2.CV_64F)).mean()))
            st.write("**Median heatmap value:**", int(np.median(heat_norm)))

        # Show a small local patch analysis when user clicks
        if st.button("Show patchwise focus map (tile-based)"):
            st.info("Computing patchwise variance — this may take a second for large images.")
            patch_size = st.slider("Patch size (px)", min_value=16, max_value=256, value=64, step=16)
            h, w = gray.shape
            heat_patch = np.zeros_like(gray, dtype=np.float32)
            counts = np.zeros_like(gray, dtype=np.float32)
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    y1 = y
                    x1 = x
                    y2 = min(h, y + patch_size)
                    x2 = min(w, x + patch_size)
                    patch = gray[y1:y2, x1:x2]
                    val = variance_of_laplacian(patch)
                    heat_patch[y1:y2, x1:x2] = val
                    counts[y1:y2, x1:x2] = 1.0
            # normalize
            if heat_patch.max() > 0:
                norm_patch = np.uint8(255.0 * (heat_patch / heat_patch.max()))
            else:
                norm_patch = np.zeros_like(heat_patch, dtype=np.uint8)
            patch_colored = apply_colormap(norm_patch, cmap_choice)
            st.image(to_rgb(patch_colored), use_column_width=True)

else:
    st.info("Upload an image to begin. You can drag & drop or click to choose a file.")
    st.markdown("---")
    st.subheader("How it works")
    st.write(
        "The app computes the variance of the Laplacian (a focus measure). Low variance = few high-frequency edges = blurry. ``threshold`` controls when an image is flagged as blurry. The heatmap visualizes absolute Laplacian energy per pixel."
    )
    st.write("Tips: For portraits or low-contrast images, consider visually inspecting the heatmap — some images with low contrast can look 'blurry' to the metric even if they're intentionally soft.")

# Footer
st.markdown("---")
st.caption("Built with OpenCV, NumPy and Streamlit — Laplacian-based blur detection and visual diagnostics.")
