
import numpy as np
import scipy as sp
import cv2
import random
import os
import scipy.fftpack as fftpack
import scipy.sparse.linalg as lg
from tempfile import mkdtemp
e_dat = os.path.join(mkdtemp(), 'e.dat')
t_dat = os.path.join(mkdtemp(), 't.dat')

IMG_TEST_FOLDER = "images/output/test/"

MAX_SIZE = 200

class Npr:
    def __init__(self, images, name="", alphas=[0.0, 0.25, 0.5, 0.75, 1.0]):
        self.alphas = alphas
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
        self.max_color, self.max_gray = self.get_max_imgs()

    def get_max_imgs(self):
        max_color = np.max(self.images, axis=0)
        max_gray = np.max(self.gray_imgs, axis=0)
        max_gray[max_gray <= 0.] = 1.
        # cv2.imwrite(os.path.join(self.test_folder, 'max_color.jpg'), max_color)
        # cv2.imwrite(os.path.join(self.test_folder, 'max_gray.jpg'), max_gray)
        return max_color, max_gray

    def get_ratio_imgs(self):
        median_gray = np.median(self.gray_imgs, axis=0)
        median_gray[median_gray <= 0.] = 1.
        ratios_max = self.gray_imgs / self.max_gray
        ratios_spec = self.gray_imgs / median_gray
        ratios_spec[ratios_spec > 1.] = 1.
        m = np.argmin([np.sum(ratios_max), np.sum(ratios_spec)])
        if m == 0:
            ratios = ratios_max
        else:
            ratios = ratios_spec
        # sum_ratios = np.sum(ratios, axis=0)
        # sum_ratios[sum_ratios > 1.] = 1.
        # cv2.imwrite(os.path.join(self.test_folder, 'sum_ratios.jpg'), sum_ratios * 255.)
        # for i, r in enumerate(ratios):
        #     cv2.imwrite(os.path.join(self.test_folder, 'ratio-{}.jpg'.format(self.img_names[i])), r * 255.)
        return ratios

    def detect_depth_edges(self, ratio_imgs):
        sobels = []
        lapls = []
        sobel_operator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        for i, ratio in enumerate(ratio_imgs):
            if self.img_names[i] == "up" or self.img_names[i] == "down":
                sobelY = cv2.filter2D(ratio, cv2.CV_64F, sobel_operator)
                sobels.append(sobelY)
            else:
                sobelX = cv2.filter2D(ratio, cv2.CV_64F, sobel_operator.T)
                sobels.append(sobelX)
            lapl_op = np.array([[0,1,0],[1,-4,1],[0,1,0]])
            lapl = cv2.filter2D(ratio, cv2.CV_64F, lapl_op)
            lapls.append(lapl)

        # all_lapls = np.max(lapls, axis=0)
        # cv2.imwrite(os.path.join(self.test_folder, 'laplaces.jpg'), all_lapls*255)
        # all_sobels = np.max(sobels, axis=0)
        # cv2.imwrite(os.path.join(self.test_folder, 'sobels.jpg'), all_sobels*255)

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
        # edge_map = np.max(silhouettes, axis=0)
        # cv2.imwrite(os.path.join(self.test_folder, 'edge_map.jpg'), edge_map*255)
        edge_mask = np.zeros(silhouettes[0].shape)

        for s in silhouettes:
            # cv2.imwrite(os.path.join(self.test_folder, 'sil.jpg'), s * 255.)
            # gauss = cv2.GaussianBlur(s, (3, 3), 0)
            # cv2.imwrite(os.path.join(self.test_folder, 'gauss.jpg'), gauss * 255.)
            kernel = cv2.getGaussianKernel(5, -1, cv2.CV_64F)
            sep = cv2.sepFilter2D(s, cv2.CV_64F, kernel, kernel)
            # cv2.imwrite(os.path.join(self.test_folder, 'sep.jpg'), sep*255.)
            s_edge = np.zeros(s.shape)
            s_edge[sep > 0.4] = 1.
            edge_mask = s_edge + edge_mask
            edge_mask[edge_mask >= 1.] = 1
            # cv2.imwrite(os.path.join(self.test_folder, 'edges.jpg'), edge_mask * 255.)
        # cv2.imwrite(os.path.join(self.test_folder, 'edge_mask.jpg'), edge_mask * 255.)
        return edge_mask

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

    def render_signed_edges(self, ratio_imgs, edges):
        signed = np.ones(edges.shape, dtype=np.float64) * 0.5
        signed[edges == 1.] = 0.
        thresh = -0.15
        r_edges = np.zeros(ratio_imgs[0].shape)
        n_att = 6
        scores = np.zeros(signed.shape)
        for i, ratio in enumerate(ratio_imgs):
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
                bs = cv2.GaussianBlur(signed, (3,3), sigmaX=1, sigmaY=0)
            else:
                bs = cv2.GaussianBlur(signed, (3,3), sigmaX=0, sigmaY=1)
            # cv2.imwrite(os.path.join(self.test_folder, 'bs.jpg'), bs * 255.)
            signed[edges == 1.] = 0.
            signed[bs >= 0.51] = 1.
            # cv2.imwrite(os.path.join(self.test_folder, 'signed4.jpg'), signed * 255.)
            # cv2.imwrite(os.path.join(self.test_folder, 'r_edges2.jpg'), r_edges*255.)

        r_img = np.copy(r_edges)
        r_img[r_img > 1.] = 1.
        # cv2.imwrite(os.path.join(self.test_folder, 'signed3.jpg'), signed * 255.)
        # cv2.imwrite(os.path.join(self.test_folder, 'r_edges.jpg'), r_img*255.)
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
        # cv2.imwrite(os.path.join(self.test_folder, 'max_gray_sobel.jpg'), sobel)
        texture_mask[sobel > 20] = 1.
        # ed = cv2.boxFilter(edges, -1, (9, 9))
        # ed[ed > 0.] = 1.
        # texture_mask[ed == 1.] = 0.
        # cv2.imwrite(os.path.join(self.test_folder, 'texture_mask.jpg'), texture_mask * 255.)
        canny = cv2.Canny(np.uint8(max_gray), 0, 250)
        # cv2.imwrite(os.path.join(self.test_folder, 'canny.jpg'), canny)
        return texture_mask, canny

    # Ratio of the distance field of texture pixels
    # by the distance field of depth edge pixels.
    # The distance field value at a pixel is the
    # Euclidean distance to the nearest (texture
    # or depth) edge pixel.
    def distance_ratio(self, f, edges, textures):
        featureless_pyr = self.reduce(f)
        edge_pyr = self.reduce(edges)
        texture_pyr = self.reduce(textures)
        idx = len(featureless_pyr) - 1
        for i, fp in enumerate(featureless_pyr):
            if np.sum(np.array(fp.shape) - MAX_SIZE) < 0:
                idx = i
                break
        f1 = featureless_pyr[idx]
        edge_mask = edge_pyr[idx]
        texture_mask = texture_pyr[idx]
        # cv2.imwrite(os.path.join(self.test_folder, 'f1.jpg'), f1*255.)
        featureless = np.array(np.where(f1 == 0.)).T
        edge_dist = np.zeros(edge_mask.shape)
        edgew = np.where(edge_mask > 0.)
        ew = np.array(edgew).T
        texture_dist = np.zeros(texture_mask.shape)
        texturew = np.where(texture_mask > 0.)
        tw = np.array(texturew).T
        lf = len(featureless)
        es = np.subtract(ew, featureless[:, np.newaxis])
        ts = np.subtract(tw, featureless[:, np.newaxis])
        e_euc = np.min(np.sqrt(np.sum(np.square(es), axis=2)), axis=1)
        t_euc = np.min(np.sqrt(np.sum(np.square(ts), axis=2)), axis=1)
        for i in xrange(lf):
            pt = featureless[i]
            edge_dist[pt[0]][pt[1]] = e_euc[i]
            texture_dist[pt[0]][pt[1]] = t_euc[i]

        edge_dist = self.expand(edge_dist)[idx]
        text_dist = self.expand(texture_dist)[idx]
        edge_dist[edge_dist == 0.] = 1.
        dist_ratio = text_dist / edge_dist
        dist_ratio[edges == 1.] = 0.
        dist_ratio[textures == 1.] = 0.
        return dist_ratio

    def create_attentuation_map(self, r_edges, edges, signed):
        edges_m = np.zeros(edges.shape)
        edges_m[edges == 0.] = 1.
        kernel = cv2.getGaussianKernel(13, -1, cv2.CV_64F)
        sep = cv2.sepFilter2D(edges_m * 255., cv2.CV_64F, kernel, kernel)
        # cv2.imwrite(os.path.join(self.test_folder, 'sep_cam.jpg'), sep)
        d = cv2.filter2D(edges_m*255., -1, kernel)
        # cv2.imwrite(os.path.join(self.test_folder, 'd_cam.jpg'), d)
        gb = cv2.GaussianBlur(edges_m*255., (5,5), -1)
        # cv2.imwrite(os.path.join(self.test_folder, 'gb_cam.jpg'), gb)
        return d

    def colorize(self, r_edges, edges, signed):
        max_color = np.copy(self.max_color)
        max_gray = np.copy(self.max_gray)
        b = np.copy(self.max_color)

        ## Build Gamma mask
        sobel_y = cv2.Sobel(max_color, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(max_color, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        color_grad = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        # cv2.imwrite(os.path.join(self.test_folder, 'color_grad.jpg'), color_grad)

        texture_mask, canny = self.get_texture_pixels(max_gray, edges)
        f = np.zeros(texture_mask.shape)
        f[texture_mask != 0.] = 1.
        text_dist = cv2.distanceTransform(np.uint8(f), cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
        f = np.zeros(texture_mask.shape)
        f[edges != 0.] = 1.
        edge_dist = cv2.distanceTransform(np.uint8(f), cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
        edge_dist[edge_dist == 0.] = 1.
        dist_ratio = text_dist / edge_dist
        colorized_imgs = []
        # X = self.poisson_solver(max_gray, max_gray)
        # cv2.imwrite(os.path.join(self.test_folder, 'X.jpg'), X * 255.)
        for alpha in self.alphas:
            gamma = np.zeros(color_grad.shape)
            distw = np.where(dist_ratio != 0.)
            for i in xrange(len(distw[0])):
                gamma[distw[0][i], distw[1][i], :] = dist_ratio[distw[0][i]][distw[1][i]] * alpha
            gamma[texture_mask == 1.] = alpha
            gamma[edges == 1.] = 1.
            G = np.power(color_grad, gamma)
            I_prime = np.abs(color_grad - G)
            # cv2.imwrite(os.path.join(self.test_folder, 'iprime_{}.jpg'.format(alpha)), I_prime)
            # a = I_prime / np.average(I_prime, axis=2)[:, :, np.newaxis]
            s = max_color - I_prime
            # cv2.imwrite(os.path.join(self.test_folder, 'iprime2_{}.jpg'.format(alpha)), s)
            I_norm = cv2.normalize(s, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
            I = np.copy(I_norm-20.)
            # cv2.imwrite(os.path.join(self.test_folder, 'inorm_{}.jpg'.format(alpha)), I_norm)
            I[edges == 1.] = 0.
            # cv2.imwrite(os.path.join(self.test_folder, 'colorized_{}.jpg'.format(alpha)), I)
            colorized_imgs.append(I)

        mean_shift = self.segment(max_color)
        mean_shift[edges == 1.] = 0.
        # cv2.imwrite(os.path.join(self.test_folder, 'mean_shift.jpg'), mean_shift)
        for i in range(2):
            b = cv2.GaussianBlur(b, (13, 13), 0)
        b[edges == 1.] = 0.
        # cv2.imwrite(os.path.join(self.test_folder, 'b.jpg'), b)
        return colorized_imgs, mean_shift, b

    def poisson_solver(self, G, I):
        div_G = cv2.Laplacian(G, cv2.CV_64F, borderType=cv2.BORDER_REFLECT_101)
        G_pyr = self.reduce(div_G)
        dst_pyr = self.reduce(I)
        idx = len(G_pyr) - 1
        for i, fp in enumerate(G_pyr):
            if np.sum(np.array(fp.shape) - MAX_SIZE) < 0:
                idx = i
                break
        h, w = G_pyr[idx].shape
        M = (np.eye(h*w, h*w) * 4) + (np.eye(h*w, h*w, k=1)*-1) + (np.eye(h*w, h*w, k=-1)*-1)
        b = np.zeros(h*w)
        count = 0
        H = h - 1
        W = w - 1
        for r in xrange(0, H):
            for c in xrange(0, W):
                i = ((r-1) * H) + c
                if r == 0:
                    b[count] = b[count] + dst_pyr[idx][r+1][c-1]
                if r == H:
                    b[count] = b[count] + dst_pyr[idx][r+1][c]

                if c == 0:
                    b[count] = b[count] + dst_pyr[idx][r][c-1]
                if c == W:
                    b[count] = b[count] + dst_pyr[idx][r][c+1]

                xv = c - 1
                yv = r - 1

                v = G_pyr[idx][yv][xv]
                b[count] = b[count] + v
                count += 1

        X = lg.bicg(M, b.T)
        im = np.reshape(X[0], (h, w))
        dst = self.expand(im)[idx]
        return np.abs(dst[:G.shape[0], :G.shape[1]])

    def handle_specularities(self):
        intensity_grads = []
        max_color, max_gray = self.get_max_imgs()
        for img in self.images:
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            grad = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
            intensity_grads.append(grad)
        int_grays = []
        for img in self.gray_imgs:
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            grad = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
            int_grays.append(grad)
        med = np.median(np.array(intensity_grads, dtype=np.float64), axis=0)
        med_gray = np.median(np.array(int_grays, dtype=np.float64), axis=0)
        # cv2.imwrite(os.path.join(self.test_folder, 'med.jpg'), med)
        # cv2.imwrite(os.path.join(self.test_folder, 'med_gray.jpg'), med)

        # lapl = cv2.Laplacian(max_gray, cv2.CV_64F, borderType=cv2.BORDER_REFLECT_101)
        # cv2.imwrite(os.path.join(self.test_folder, 'lapl_spec.jpg'), lapl)
        # dst = fftpack.dst(lapl)
        # dst_t = fftpack.dst(dst.T)
        # xv, yv = np.meshgrid(np.arange(med.shape[0]), np.arange(med.shape[1]))
        # d = (2 * np.cos(np.divide(np.pi * xv, med.shape[0]-1))) + (2 * np.cos(np.divide(np.pi * yv, med.shape[1]-1)))
        # f = dst_t / d
        # idst = fftpack.idst(f)
        # img_idst = fftpack.idst(idst.T)
        # cv2.imwrite(os.path.join(self.test_folder, 'idst.jpg'), img_idst)

        recon = np.abs(np.subtract(max_color, med))
        # cv2.imwrite(os.path.join(self.test_folder, 'recon.jpg'), recon)
        recon_gray = cv2.cvtColor(np.uint8(recon), cv2.COLOR_BGR2GRAY)
        recon_gray[recon_gray <= 0.] = 1.
        med_mask = np.zeros(med_gray.shape)
        med_mask[med_gray > 100] = 1.
        # cv2.imwrite(os.path.join(self.test_folder, 'med_mask.jpg'), med_mask * 255.)
        # cv2.imwrite(os.path.join(self.test_folder, 'med_g1.jpg'), np.median(self.gray_imgs, axis=0))
        gp = self.gauss_pyr(max_gray, np.median(self.gray_imgs, axis=0), med_mask)
        # cv2.imwrite(os.path.join(self.test_folder, 'gp.jpg'), gp)
        return recon, recon_gray, gp

    def gauss_pyr(self, img1, img2, mask):
        r = 3
        I1 = img1.copy()
        I2 = img2.copy()
        m = mask.copy()
        gp1 = [I1]
        gp2 = [I2]
        gp_mask = [mask]
        for i in xrange(r):
            I1 = cv2.pyrDown(I1)
            I2 = cv2.pyrDown(I2)
            m = cv2.pyrDown(m)
            m[m > 0.] = 1.
            gp1.append(I1)
            gp2.append(I2)
            gp_mask.append(m)

        gp_mask.reverse()

        lp1 = [gp1[r-1]]
        lp2 = [gp2[r-1]]
        for i in xrange(r-1, 0, -1):
            GE1 = cv2.pyrUp(gp1[i])
            GE2 = cv2.pyrUp(gp2[i])
            h, w = gp1[i-1].shape
            GE1 = GE1[:h, :w]
            GE2 = GE2[:h, :w]
            L1 = np.subtract(gp2[i - 1], GE1)
            L2 = np.subtract(gp1[i - 1], GE2)
            lp1.append(L1)
            lp2.append(L2)

        o = []
        for i in xrange(r):
            f1 = lp1[i].copy()
            f2 = lp2[i].copy()
            f1[gp_mask[i+1] == 1.] = 0.
            f2[gp_mask[i+1] == 0.] = 0.
            ff = f1 + f2
            o.append(ff)

        out = o[0]
        for i in xrange(1, r):
            out = cv2.pyrUp(out)
            h, w = o[i].shape
            out = out[:h, :w] + o[i]
        return out

    def gauss_pyr_c(self, img1, img2, mask):
        r = 3
        I1 = img1.copy()
        I2 = img2.copy()
        m = mask.copy()
        gp1 = [I1]
        gp2 = [I2]
        gp_mask = [mask]
        for i in xrange(r):
            I1 = cv2.pyrDown(I1)
            I2 = cv2.pyrDown(I2)
            m = cv2.pyrDown(m)
            m[m > 0.] = 1.
            gp1.append(I1)
            gp2.append(I2)
            gp_mask.append(m)

        gp_mask.reverse()

        lp1 = [gp1[r-1]]
        lp2 = [gp2[r-1]]
        for i in xrange(r-1, 0, -1):
            GE1 = cv2.pyrUp(gp1[i])
            GE2 = cv2.pyrUp(gp2[i])
            h, w, _ = gp1[i-1].shape
            GE1 = GE1[:h, :w, :]
            GE2 = GE2[:h, :w, :]
            L1 = np.subtract(gp2[i - 1], GE1)
            L2 = np.subtract(gp1[i - 1], GE2)
            lp1.append(L1)
            lp2.append(L2)

        o = []
        for i in xrange(r):
            f1 = lp1[i].copy()
            f2 = lp2[i].copy()
            f1[gp_mask[i+1] == 1.] = 0.
            f2[gp_mask[i+1] == 0.] = 0.
            ff = f1 + f2
            o.append(ff)

        out = o[0]
        for i in xrange(1, r):
            out = cv2.pyrUp(out)
            h, w, _ = o[i].shape
            out = out[:h, :w, :] + o[i]
        return out

    def reduce(self, img):
        i_c = img.copy()
        out = [i_c]
        for i in xrange(6):
            i_c = cv2.pyrDown(i_c)
            out.append(i_c)
        return out

    def expand(self, img):
        i_c = img.copy()
        out = [i_c]
        for i in xrange(6):
            i_c = cv2.pyrUp(i_c)
            out.append(i_c)
        return out

    def segment(self, img):
        p = cv2.pyrMeanShiftFiltering(np.uint8(img), 10, 20, 5)
        return p


    def run(self):
        results = {}

        # Get Ratio Images
        ratio_imgs = self.get_ratio_imgs()
        for i, r in enumerate(ratio_imgs):
            n = 'ratio-{}.jpg'.format(self.img_names[i])
            results[n] = r * 255.

        # Get Depth Edges
        edge_mask = self.detect_depth_edges(ratio_imgs)
        results['edges'] = edge_mask * 255.

        # Get Signed Edges
        ratio_edges, _, signed_edges = self.render_signed_edges(ratio_imgs, edge_mask)
        results['ratio_edges'] = ratio_edges * 255.
        results['signed_edges'] = signed_edges * 255.

        # Colorize
        colorized_imgs, mean, b = self.colorize(ratio_edges, edge_mask, signed_edges)

        for i, c in enumerate(colorized_imgs):
            n = 'colorized_{}'.format(self.alphas[i])
            results[n] = c
        results['colorized_mean_filtered'] = mean
        results['colorized_blurred'] = b

        return results