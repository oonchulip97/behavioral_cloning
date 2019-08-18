import matplotlib.pyplot as plt
import numpy as np

# Visualize images from center, left and right cameras
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
img1 = plt.imread("./writeup_images/center.jpg")
img2 = plt.imread("./writeup_images/left.jpg")
img3 = plt.imread("./writeup_images/right.jpg")
f.tight_layout()
ax1.imshow(img1)
ax1.axis("off")
ax1.set_title("Center", fontsize=30)
ax2.imshow(img2)
ax2.axis("off")
ax2.set_title("Left", fontsize=30)
ax3.imshow(img3)
ax3.axis("off")
ax3.set_title("Right", fontsize=30)
plt.savefig("./writeup_images/three_cameras.png", bbox_inches='tight')

# Visualize flipped image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
img1 = plt.imread("./writeup_images/center.jpg")
img2 = np.fliplr(img1)
f.tight_layout()
ax1.imshow(img1)
ax1.axis("off")
ax1.set_title("Original", fontsize=30)
ax2.imshow(img2)
ax2.axis("off")
ax2.set_title("Flipped", fontsize=30)
plt.savefig("./writeup_images/flipped.png", bbox_inches='tight')

# Visualize cropped image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
img1 = plt.imread("./writeup_images/center.jpg")
img2 = img1[50:140,:,:]
f.tight_layout()
ax1.imshow(img1)
ax1.axis("off")
ax1.set_title("Original", fontsize=30)
ax2.imshow(img2)
ax2.axis("off")
ax2.set_title("Cropped", fontsize=30)
plt.savefig("./writeup_images/cropped.png", bbox_inches='tight')