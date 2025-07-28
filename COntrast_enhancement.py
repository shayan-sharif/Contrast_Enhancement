"""
Contrast Enhancement using Histogram Equalization, Stretching, and CLAHE
Author: Your Name
GitHub: https://github.com/your-username
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== Step 1: Read and display original image =====================
img = cv2.imread("img1.jpg")
if img is None:
    raise FileNotFoundError("Image file not found. Make sure 'img1.jpg' exists.")

cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===================== Step 2: Plot Histogram =====================
hist, bins = np.histogram(img.flatten(), bins=64, range=[0,256])
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(6, 4))
plt.bar(bin_centers, hist, width=4)
plt.title('Original Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================== Step 3: Convert to Grayscale =====================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===================== Step 4: Histogram Equalization =====================
equalized = cv2.equalizeHist(gray)

cv2.imshow('Histogram Equalized', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===================== Step 5: Contrast Stretching =====================
min_val = np.min(gray)
max_val = np.max(gray)

print(f"Min Pixel Value: {min_val}, Max Pixel Value: {max_val}")

stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)

cv2.imshow('Contrast Stretched', stretched)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===================== Step 6: CLAHE (Adaptive Histogram Equalization) =====================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(gray)

cv2.imshow('CLAHE Enhanced', clahe_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
