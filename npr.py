
import numpy as np
import scipy as sp
import cv2
import random
import os

IMG_TEST_FOLDER = "images/output/test"

class Npr:
    def __init__(self, images, alpha=1.0):
        self.alpha = alpha
        self.ambient = None
        if 'ambient' in images:
            self.ambient = images['ambient']
            images.pop('ambient', None)
        imgs_arr = images.values()
        self.img_names = images.keys()
        self.name_idxs = dict((i, x) for i, x in enumerate(images.keys()))
        self.images = np.float64(imgs_arr)
        self.gray_imgs = np.array([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs_arr], dtype=np.float64)
        self.means = np.array([np.mean(g) for g in self.gray_imgs])
        self.mean = np.mean(self.gray_imgs)
        self.normalized = np.array([
            np.multiply(self.gray_imgs[i], np.divide(self.mean, self.means[i]))
            for i in range(self.gray_imgs.shape[0])
        ])

    def get_max_imgs(self):
        max_color = np.max(self.images, axis=0)
        max_gray = np.max(self.gray_imgs, axis=0)
        cv2.imwrite(os.path.join(IMG_TEST_FOLDER, 'max_color.jpg'), max_color)
        cv2.imwrite(os.path.join(IMG_TEST_FOLDER, 'max_gray.jpg'), max_gray)
        return max_color, max_gray

    def get_ratio_imgs(self):
        _, max_gray = self.get_max_imgs()
        ratios = self.gray_imgs / max_gray
        for i, r in enumerate(ratios):
            cv2.imwrite(os.path.join(IMG_TEST_FOLDER, 'ratio-{}.jpg'.format(self.img_names[i])), r * 255.)
        return ratios

    def run_canny(self):
        cv2.Canny()

    def detect_depth_edges(self):
        ratios = self.get_ratio_imgs()
        sobels = []
        lapls = []
        sobel_operator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        for i, ratio in enumerate(ratios):
            if self.img_names[i] == "up" or self.img_names[i] == "down":
                sobel = cv2.filter2D(ratio, cv2.CV_64F, sobel_operator)
                sobelY = cv2.Sobel(ratio, cv2.CV_64F, 0, 1, ksize=3)
                # sobel = cv2.convertScaleAbs(sobelY)
                # sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
            else:
                sobel = cv2.filter2D(ratio, cv2.CV_64F, sobel_operator.T)
                sobelX = cv2.Sobel(ratio, cv2.CV_64F, 1, 0, ksize=3)
                # sobel = cv2.convertScaleAbs(sobelX)
                # sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
            sobels.append(sobel)
            lapl_op = np.array([[0,1,0],[1,-4,1],[0,1,0]])
            lapl = cv2.filter2D(ratio, cv2.CV_64F, lapl_op)
            lapls.append(lapl)

        all_lapls = np.max(lapls, axis=0)
        cv2.imwrite(os.path.join(IMG_TEST_FOLDER, 'laplaces.jpg'), all_lapls*255)
        all_sobels = np.max(sobels, axis=0)
        cv2.imwrite(os.path.join(IMG_TEST_FOLDER, 'sobels.jpg'), all_sobels*255)

        silhouettes = []
        for i, edge in enumerate(sobels):
            edge_mask = np.zeros(edge.shape)
            if self.img_names[i] == "up" or self.img_names[i] == "left":
                edge_mask[edge > 0.] = 1.
                silhouette = edge * edge_mask
            else:
                edge_mask[edge < 0.] = 1.
                silhouette = np.abs(edge * edge_mask)
            silhouettes.append(silhouette)
        silhouettes = np.float64(silhouettes)
        edge_map = np.max(silhouettes, axis=0)
        cv2.imwrite(os.path.join(IMG_TEST_FOLDER, 'edge_map.jpg'), edge_map*255)
        low_thresh = 0.5
        hi_thresh = 1.0
        edges = np.zeros(silhouettes[0].shape)
        for s in silhouettes:
            s_edge = np.zeros(s.shape)
            s_edge[s > 0.8] = 1.
            edges = s_edge + edges
            edges[edges >= 1.] = 1
            cv2.imwrite(os.path.join(IMG_TEST_FOLDER, 'edges.jpg'), edges*255.)
        return edges

    def create_mask_image(self):
        max_color, _ = self.get_max_imgs()

        return NotImplementedError

    def create_attentuation_map(self):
        edges = self.detect_depth_edges()

        return NotImplementedError

    def create_attenuated_image(self):
        return NotImplementedError

    def colorize(self):
        return NotImplementedError

    def run(self):
        self.detect_depth_edges()