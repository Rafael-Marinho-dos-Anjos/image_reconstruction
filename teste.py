
import cv2
import numpy as np
from torch.nn import Sigmoid, ReLU
from torch import FloatTensor


sig = Sigmoid()
rel = ReLU()
img = cv2.imread("gt/2.jpg")

b, g, r = np.transpose(img, (2, 0, 1))

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

cv2.imshow("image", img)
cv2.waitKey(0)

b = (b - np.mean(b)) / np.std(b)
g = (g - np.mean(g)) / np.std(g)
r = (r - np.mean(r)) / np.std(r)

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

img = np.transpose(np.array([b, g, r]), (1, 2, 0))

cv2.imshow("image", img)
cv2.waitKey(0)

b = sig(FloatTensor(b)).numpy()
g = sig(FloatTensor(g)).numpy()
r = sig(FloatTensor(r)).numpy()

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

img = np.transpose(np.array([b, g, r]), (1, 2, 0))

cv2.imshow("image", img)
cv2.waitKey(0)
