# -*- coding: utf-8 -*-

from ncs import NCS
import cv2

movidius = NCS()
img = cv2.imread('photos/sample.jpg')

movidius.infer_image(img)

movidius.close_ncs_device()