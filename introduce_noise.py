from os import listdir
import cv2
from random import randint
from copy import deepcopy
from numpy import zeros


EXAMPLES_PER_IMAGE = 10
MIN_WINDOW_SIZE = 0.1
MAX_WINDOW_SIZE = 0.2

images = listdir("gt")

for path in images:
    image = cv2.imread("gt\\{}".format(path))

    for i in range(EXAMPLES_PER_IMAGE):
        window_shape = [randint(360*MIN_WINDOW_SIZE, 360*MAX_WINDOW_SIZE) for i in range(2)]
        window_loc   = [randint(0, 359-dim) for dim in window_shape]
        img = deepcopy(image)
        img[window_loc[0]: window_loc[0]+window_shape[0], window_loc[1]: window_loc[1]+window_shape[1]] = zeros((*window_shape, 3))
        cv2.imwrite("data\\{}".format(path[:-4]+"_"+str(i)+path[-4:]), img)
