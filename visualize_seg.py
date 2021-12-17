import cv2
from skimage import io, color
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# image = cv2.imread("/home/rafay_veeve/Desktop/Veeve/galactus/HO3D_v3/train/MC1/rgb/0000.jpg")
# gt = cv2.imread("/home/rafay_veeve/Desktop/Veeve/galactus/HO3D_v3/train/MC1/segmentations/0000.png")
image = io.imread("/home/rafay_veeve/Desktop/Veeve/galactus/HO3D_v3/train/MC1/rgb/0500.jpg")
# gt = io.imread("/home/rafay_veeve/Desktop/Veeve/galactus/HO3D_v3/train/MC1/segmentations/0500.png")
gt = Image.open("/home/rafay_veeve/Desktop/Veeve/galactus/HO3D_v3/train/MC1/segmentations/0500.png")

# seg = np.zeros(image.shape[:2])
# seg[np.where(np.all(gt == (255, 0, 0), axis=-1))] = 1
# seg[np.where(np.all(gt == (0, 0, 255), axis=-1))] = 2


print(np.array(gt)[229, 331])
plt.imshow(gt)
plt.show()
cv2.waitKey()


plt.imshow(image)
plt.show()
cv2.waitKey()

io.imshow(color.label2rgb(np.array(gt), image, colors=[(255, 0, 0), (0, 0, 255)],alpha=0.01, bg_label=0, bg_color=None))
plt.show()
cv2.waitKey()


