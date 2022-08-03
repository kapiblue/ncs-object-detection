import NCS
import cv2

ncs = NCS()
img = cv2.imread('photos/sample.jpg')

ncs.infer_image(img)