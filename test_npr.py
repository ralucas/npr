import cv2
import numpy as np
import scipy as sp
import unittest

import os
import npr as NPR

IMG_SRC_FOLDER = "images/source"

def get_images(folder_name):
    images = {}
    for file in os.listdir(os.path.join(IMG_SRC_FOLDER, folder_name)):
        image = cv2.imread(os.path.join(IMG_SRC_FOLDER, folder_name, file))
        name = file.split(".")[0]
        images[name] = image
    return images

class NprTest(unittest.TestCase):

    def setUp(self):
        images = {
            'engine': get_images('engine'),
            'flower': get_images('flower'),
            'my_flower': get_images('my_flower')
        }

        self.images = images

    def test_depth_edge_detection(self):
        npr = NPR.Npr(self.images['my_flower'])
        dde = npr.detect_depth_edges()
        self.assertTrue(dde)

if __name__ == '__main__':
    unittest.main()

