import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def sobel_edge_detection(image):
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    edges = np.hypot(dx, dy)  # Magnitude of gradient
    edges = (edges / np.max(edges) * 255).astype(np.uint8)  # Normalize to 0-255
    return edges

def thresholding(image, threshold=128):
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binary_image

# Read the image
input_path = "image.png"  # Replace with your image path
image = imageio.imread(input_path, mode='F').astype(np.uint8)  # Use mode='F'

edges = sobel_edge_detection(image)

threshold = 128  # You can adjust the threshold value
segmented_image = thresholding(edges, threshold)

# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Edges (Sobel)")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Segmented Image")
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()