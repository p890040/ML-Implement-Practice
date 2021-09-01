#%% Load and preprocess
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('Images/img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))
fig = plt.figure()
plt.imshow(img)

img_vector = img.reshape(-1,3).astype(np.float32)

#%% OpenCV K-means filter, K==2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
K = 2
_, classes, centers = cv2.kmeans(img_vector, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = centers.astype(np.uint8)
img_segment = centers[classes]
img_segment = img_segment.reshape(img.shape)

classes_segment = classes.reshape(img.shape[0], img.shape[1])

fig = plt.figure()
plt.imshow(img_segment)

#%% OpenCV K-means filter, K==3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
K = 3
_, classes, centers = cv2.kmeans(img_vector, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = centers.astype(np.uint8)
img_segment = centers[classes]
img_segment = img_segment.reshape(img.shape)

classes_segment = classes.reshape(img.shape[0], img.shape[1])

fig = plt.figure()
plt.imshow(img_segment)


