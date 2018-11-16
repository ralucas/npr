import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import npr

IMG_FOLDER = "images/source"

class NprTest(unittest.TestCase):

    def setUp(self):
        images = [cv2.imread(path.join(IMG_FOLDER, "sample-00.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-01.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-02.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-03.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-04.png")),
                  cv2.imread(path.join(IMG_FOLDER, "sample-05.png"))]

        if not all([im is not None for im in images]):
            raise IOError("Error, one or more sample images not found.")

        self.images = images

if __name__ == '__main__':
    unittest.main()

