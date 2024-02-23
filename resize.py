from os import listdir
import cv2


images = listdir("original_images")

for i, path in enumerate(images):

    img = cv2.imread(r"original_images\\" + path)

    if img.shape[0] > img.shape[1]:
        start = (img.shape[0] - img.shape[1]) // 2
        img = img[start: start + img.shape[1], :, :]
    
    elif img.shape[1] > img.shape[0]:
        start = (img.shape[1] - img.shape[0]) // 2
        img = img[:, start: start + img.shape[0], :]

    img = cv2.resize(img, (360, 360), cv2.INTER_AREA)

    cv2.imwrite("gt\\{}.jpg".format(i), img)
