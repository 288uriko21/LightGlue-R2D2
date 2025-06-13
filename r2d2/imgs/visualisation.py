import cv2
import numpy as np
import matplotlib.pyplot as plt

data = np.load('brooklyn.png.r2d2')
img = cv2.imread("brooklyn.png")

print(data.files)
keyp = data['keypoints']
desk = data['descriptors']

print(keyp)

for point in keyp:
    cv2.circle(img, (int(point[0]), int(point[1])), 10, (0, 255, 0), 3)

plt.imshow(img)
plt.savefig('result.png')
