import cv2
import numpy as np
import sys

import os
import errno

from os import path

import npr

IMG_SRC_FOLDER = 'images/source'
IMG_OUT_FOLDER = 'images/output'

def get_images(folder_name):
    images = {}
    for file in os.listdir(os.path.join(IMG_SRC_FOLDER, folder_name)):
        image = cv2.imread(os.path.join(IMG_SRC_FOLDER, folder_name, file))
        name = file.split(".")[0]
        images[name] = image
    return images

def run(name):
    images = get_images(name)
    n = npr.Npr(images, name=name)
    n.run()

if __name__ == "__main__":
    dir = sys.argv[1]
    if not dir:
        print "Missing directory argument: \n\t python main.py [directory]"
    run(dir)

