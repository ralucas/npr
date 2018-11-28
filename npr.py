
import numpy as np
import scipy as sp
import cv2
import random
import os

IMG_TEST_FOLDER = "images/output/test/"

class Npr:
    def __init__(self, images, name="", alpha=1.0):
        self.alpha = alpha
        self.name = name
        self.test_folder = IMG_TEST_FOLDER + name
        try:
            os.mkdir(self.test_folder)
        except:
            print "dir exists"
        self.ambient = None
        if 'ambient' in images:
            self.ambient = images['ambient']
            self.amb_gray = cv2.cvtColor(self.ambient, cv2.COLOR_BGR2GRAY)
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
        # cv2.imwrite(os.path.join(self.test_folder, 'max_255.jpg'), np.ones(self.gray_imgs[0].shape)*255.)
        # cv2.imwrite(os.path.join(self.test_folder, 'max_0.jpg'), np.zeros(self.gray_imgs[0].shape))

    def get_max_imgs(self):
        max_color = np.max(self.images, axis=0)
        max_gray = np.max(self.gray_imgs, axis=0)
        max_gray[max_gray <= 0.] = 1.
        cv2.imwrite(os.path.join(self.test_folder, 'max_color.jpg'), max_color)
        cv2.imwrite(os.path.join(self.test_folder, 'max_gray.jpg'), max_gray)
        return max_color, max_gray

    def get_ratio_imgs(self):
        self.max_color, self.max_gray = self.get_max_imgs()
        ratios = self.gray_imgs / self.max_gray
        for i, r in enumerate(ratios):
            cv2.imwrite(os.path.join(self.test_folder, 'ratio-{}.jpg'.format(self.img_names[i])), r * 255.)
        return ratios

    # TODO: 1. Handle Specularities with Gaussian Gradient
    def detect_depth_edges(self):
        self.ratios = self.get_ratio_imgs()
        sobels = []
        lapls = []
        sobel_operator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        for i, ratio in enumerate(self.ratios):
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
        cv2.imwrite(os.path.join(self.test_folder, 'laplaces.jpg'), all_lapls*255)
        all_sobels = np.max(sobels, axis=0)
        cv2.imwrite(os.path.join(self.test_folder, 'sobels.jpg'), all_sobels*255)

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
        self.edge_map = np.max(silhouettes, axis=0)
        cv2.imwrite(os.path.join(self.test_folder, 'edge_map.jpg'), self.edge_map*255)
        low_thresh = 0.5
        hi_thresh = 1.0
        edge_mask = np.zeros(silhouettes[0].shape)

        for s in silhouettes:
            cv2.imwrite(os.path.join(self.test_folder, 'sil.jpg'), s * 255.)
            # lapl_op = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            # lapl = cv2.filter2D(s, cv2.CV_64F, lapl_op)
            # cv2.imwrite(os.path.join(self.test_folder, 'lapl.jpg'), lapl * 255.)
            # kernel = cv2.getGaussianKernel(3, 1, cv2.CV_64F)
            # gauss = cv2.filter2D(s, cv2.CV_64F, kernel)
            gauss = cv2.GaussianBlur(s, (3, 3), 0)
            cv2.imwrite(os.path.join(self.test_folder, 'gauss.jpg'), gauss * 255.)
            kernel = cv2.getGaussianKernel(5, -1, cv2.CV_64F)
            sep = cv2.sepFilter2D(s, cv2.CV_64F, kernel, kernel)
            cv2.imwrite(os.path.join(self.test_folder, 'sep.jpg'), sep*255.)
            s_edge = np.zeros(s.shape)
            s_edge[sep > 0.5] = 1.
            edge_mask = s_edge + edge_mask
            edge_mask[edge_mask >= 1.] = 1
            cv2.imwrite(os.path.join(self.test_folder, 'edges.jpg'), edge_mask * 255.)
        cv2.imwrite(os.path.join(self.test_folder, 'edge_mask.jpg'), edge_mask * 255.)
        return edge_mask

    def render_edges(self):
        self.edge_mask = self.detect_depth_edges()
        signed = np.ones(self.edge_mask.shape, dtype=np.float64) * 0.5
        signed[self.edge_mask == 1.] = 0.
        cv2.imwrite(os.path.join(self.test_folder, 'signed1.jpg'), signed * 255.)
        w = np.where(self.edge_mask == 1.)
        f = 0.
        t = 1.
        for i, ratio in enumerate(self.ratios):
            if self.img_names[i] == 'up' or self.img_names[i] == 'down':
                wu = w[0] - 1, w[1]
                wu[0][wu[0] < 0] = 0
                wd = w[0] + 1, w[1]
                wd[0][wd[0] > signed.shape[0]-1] = signed.shape[0]-1
                gu = np.greater(ratio[wu], ratio[wd])
                gu[gu == False] = f
                gu[gu == True] = t
                signed[wu] = gu
                gu[gu == f] = t
                gu[gu == t] = f
                signed[wd] = gu
            else:
                wl = w[0], w[1] - 1
                wl[1][wl[1] < 0] = 0
                wr = w[0], w[1] + 1
                wr[1][wr[1] > signed.shape[1]-1] = signed.shape[1]-1
                gu = np.greater(ratio[wl], ratio[wr])
                gu[gu == False] = f
                gu[gu == True] = t
                signed[wl] = gu
                gu[gu == f] = t
                gu[gu == t] = f
                signed[wr] = gu
        cv2.imwrite(os.path.join(self.test_folder, 'signed2.jpg'), signed * 255.)
        return signed

    def handle_edge_assign(self, r, c, signed, scores, score, dir, n_att):
        dirs = {
            'up': {
                'c': n_att,
                'r': 0,
                'dc': 0,
                'dr': -1
            },
            'down': {
                'c': n_att,
                'r': 0,
                'dc': 0,
                'dr': 1

            },
            'left': {
                'c': 0,
                'r': n_att,
                'dc': -1,
                'dr': 0
            },
            'right': {
                'c': 0,
                'r': n_att,
                'dc': 1,
                'dr': 0
            }
        }
        d = dirs[dir]
        if d['r'] != 0:
            m = np.min(scores[r-d['r']:r+d['r'], c])
        else:
            m = np.min(scores[r, c-d['c']:c+d['c']])
        if m == 0:
            for n in xrange(1, n_att+1):
                rn = d['dr'] * n
                cn = d['dc'] * n
                signed[r+rn][c+cn] = 1
            if d['r'] != 0:
                scores[r-d['r']:r+d['r'], c] = score
            else:
                scores[r, c-d['c']:c+d['c']] = score
        elif m > score:
            for n in xrange(1, n_att+1):
                rn = d['dr'] * n
                cn = d['dc'] * n
                signed[r+rn][c+cn] = 1
            for n in xrange(n_att, n_att+1):
                if rn != 0:
                    rn += 1
                if cn != 0:
                    cn += 1
                signed[r-rn][c-cn] = 0.5
            if d['r'] != 0:
                scores[r-d['r']:r+d['r'], c] = score
            else:
                scores[r, c-d['c']:c+d['c']] = score
        return signed, scores

    def render_edges2(self):
        self.ratios = self.get_ratio_imgs()
        edges = self.detect_depth_edges()
        signed = np.ones(edges.shape, dtype=np.float64) * 0.5
        signed[edges == 1.] = 0.
        thresh = -0.15
        r_edges = np.zeros(self.ratios[0].shape)
        n_att = 6
        scores = np.zeros(signed.shape)
        for i, ratio in enumerate(self.ratios):
            if self.img_names[i] == 'up':
                # start from bottom work up
                for r in xrange(ratio.shape[0]-n_att, n_att, -1):
                    for c in xrange(n_att, ratio.shape[1]-n_att):
                        score = ratio[r][c] - ratio[r-1][c]
                        if score < thresh:
                            if edges[r][c] > 0:
                                r_edges[r][c] = np.float64(i+1)
                                signed, scores = self.handle_edge_assign(r, c, signed, scores, score, self.img_names[i], n_att)
            if self.img_names[i] == 'down':
                # start from top work down
                for r in xrange(ratio.shape[0]-n_att):
                    for c in xrange(n_att, ratio.shape[1]-n_att):
                        score = ratio[r][c] - ratio[r+1][c]
                        if score < thresh:
                            if edges[r][c] > 0:
                                r_edges[r][c] = np.float64(i+1)
                                signed, scores = self.handle_edge_assign(r, c, signed, scores, score, self.img_names[i], n_att)
            if self.img_names[i] == 'right':
                # start from left work right
                for c in xrange(ratio.shape[1]-n_att):
                    for r in xrange(n_att, ratio.shape[0]-n_att):
                        score = ratio[r][c] - ratio[r][c+1]
                        if score < thresh:
                            if edges[r][c] > 0:
                                r_edges[r][c] = np.float64(i+1)
                                signed, scores = self.handle_edge_assign(r, c, signed, scores, score, self.img_names[i], n_att)
            if self.img_names[i] == 'left':
                # start from right work left
                for c in xrange(ratio.shape[1]-n_att, n_att, -1):
                    for r in xrange(n_att, ratio.shape[0]-n_att):
                        score = ratio[r][c] - ratio[r][c-1]
                        if score < thresh:
                            if edges[r][c] > 0:
                                r_edges[r][c] = np.float64(i+1)
                                signed, scores = self.handle_edge_assign(r, c, signed, scores, score, self.img_names[i], n_att)
            signed[edges == 1.] = 0.
            if self.img_names[i] == 'left' or self.img_names[i] == 'right':
                bs = cv2.GaussianBlur(signed, (3,3), sigmaX=-1, sigmaY=0)
            else:
                bs = cv2.GaussianBlur(signed, (3,3), sigmaX=0, sigmaY=-1)
            cv2.imwrite(os.path.join(self.test_folder, 'bs.jpg'), bs * 255.)
            signed[edges == 1.] = 0.
            signed[bs >= 0.51] = 1.
            cv2.imwrite(os.path.join(self.test_folder, 'signed4.jpg'), signed * 255.)
            cv2.imwrite(os.path.join(self.test_folder, 'r_edges2.jpg'), r_edges*255.)

        r_img = np.copy(r_edges)
        r_img[r_img > 1.] = 1.
        cv2.imwrite(os.path.join(self.test_folder, 'signed3.jpg'), signed * 255.)
        cv2.imwrite(os.path.join(self.test_folder, 'r_edges.jpg'), r_img*255.)
        return r_edges, edges, signed

    def create_mask_image(self):
        max_color, _ = self.get_max_imgs()
        self.r_edges, self.edges, self.signed = self.render_edges2()
        gamma_mask = np.zeros(self.signed.shape)
        gamma_mask[self.edges == 0.] = 1.
        # TODO: Create texture pixel mask/detection
        # TODO: Create distance ratio mask
        return gamma_mask

    # Defining a texture pixel as a non-edge pixel
    # that still shows a gradient
    def get_texture_pixels(self, max_gray, edges):
        texture_mask = np.zeros(max_gray.shape)
        sobel_y = cv2.Sobel(max_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(max_gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        cv2.imwrite(os.path.join(self.test_folder, 'max_gray_sobel.jpg'), sobel)
        texture_mask[(sobel < 100.) & (sobel > 35)] = 1.
        texture_mask[edges == 1.] = 0.
        cv2.imwrite(os.path.join(self.test_folder, 'texture_mask.jpg'), texture_mask * 255.)
        return texture_mask

    # Ratio of the distance field of texture pixels
    # by the distance field of depth edge pixels.
    # The distance field value at a pixel is the
    # Euclidean distance to the nearest (texture
    # or depth) edge pixel.
    def distance_ratio(self, gamma_mask, texture_mask):
        dist_mask = gamma_mask + texture_mask
        dist_mask[dist_mask > 1.] = 1.
        dist = np.zeros(dist_mask.shape)
        for i in xrange(dist.shape[0]):
            for j in xrange(dist.shape[1]):
                

    def create_attentuation_map(self):
        r_edges, edges, signed = self.render_edges2()
        edges_m = np.zeros(edges.shape)
        edges_m[edges == 0.] = 1.
        kernel = cv2.getGaussianKernel(50, -1, cv2.CV_64F)
        sep = cv2.sepFilter2D(edges_m * 255., cv2.CV_64F, kernel, kernel)
        cv2.imwrite(os.path.join(self.test_folder, 'sep_cam.jpg'), sep)
        d = cv2.filter2D(edges_m*255., -1, kernel)
        cv2.imwrite(os.path.join(self.test_folder, 'd_cam.jpg'), d)
        gb = cv2.GaussianBlur(edges_m*255., (5,5), -1)
        cv2.imwrite(os.path.join(self.test_folder, 'gb_cam.jpg'), gb)

        return d

    def create_attenuated_image(self):
        return NotImplementedError

    def colorize(self):
        # mask = self.create_mask_image()
        r_edges, edges, signed = self.render_edges2()
        max_color, max_gray = self.get_max_imgs()

        b = np.copy(max_color)
        # intensity_grad = max_color - cv2.GaussianBlur(b, (5,5), 0)
        # cv2.imwrite(os.path.join(self.test_folder, 'ig.jpg'), intensity_grad)

        ## Build Gamma mask
        sobel_y = cv2.Sobel(max_color, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(max_color, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        cv2.imwrite(os.path.join(self.test_folder, 'color_grad.jpg'), sobel)
        gamma = np.zeros(sobel.shape)
        gamma[edges == 0.] = 1.
        texture_mask = self.get_texture_pixels(max_gray, edges)
        dist = self.distance_ratio(gamma, texture_mask)
        alpha_texture = texture_mask * self.alpha

        G = np.power(sobel, gamma)
        I_prime = np.abs(max_color - G)
        cv2.imwrite(os.path.join(self.test_folder, 'iprime.jpg'), I_prime)
        I_norm = cv2.normalize(I_prime,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        I = np.copy(I_norm) * 255.
        cv2.imwrite(os.path.join(self.test_folder, 'inorm.jpg'), I)
        I[edges == 1.] = 0.
        cv2.imwrite(os.path.join(self.test_folder, 'colorized.jpg'), I)

        for i in range(2):
            b = cv2.GaussianBlur(b, (13, 13), 0)
        b[edges == 1.] = 0.
        cv2.imwrite(os.path.join(self.test_folder, 'b.jpg'), b)
        return b

    # TODO: Make this more of a pipeline
    # As well as in which intermediate images are created
    def run(self):
        self.colorize()