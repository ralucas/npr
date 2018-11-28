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
            'bone': get_images('bone'),
            'engine': get_images('engine'),
            'flower': get_images('flower'),
            'flower2': get_images('flower2'),
            'my_flower': get_images('my_flower'),
            'wine': get_images('wine')
        }

        self.images = images

    def test_depth_edge_detection(self):
        n = 'flower'
        npr = NPR.Npr(self.images[n], n)
        dde = npr.detect_depth_edges()
        self.assertTrue(dde)

    def test_render_edges(self):
        n = 'flower'
        npr = NPR.Npr(self.images[n], n)
        re = npr.render_edges()
        self.assertTrue(re)

    def test_create_attenuation_map(self):
        n = 'bone'
        npr = NPR.Npr(self.images[n], n)
        cam = npr.create_attentuation_map()
        self.assertTrue(cam)

    def test_colorize(self):
        n = 'bone'
        npr = NPR.Npr(self.images[n], n)
        cam = npr.colorize()
        self.assertTrue(cam)

if __name__ == '__main__':
    unittest.main()

