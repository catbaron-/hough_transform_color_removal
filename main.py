import cv2
import basicimage as bm
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import random
from sklearn.cluster import MeanShift, estimate_bandwidth

__author__ = 'wtq'
# apply KM when generate hough lines to reduce number of lines
# including 3 planes and 4 spheres
# coding=utf-8
# IMAGE = "input_img/teapot_ss.png"
IMAGE = "input_img/sphere_2color.png"
LOCAL = False
LOCAL = True
QUANTILE = 0.25
HOUGH_VOTE = 20
NUM_OF_HOUGH_LINE = 3
VECTOR_DIMENSION = 36
K_ = 5
HEMISPHERE_NUM = 10
COS_FOR_SKELETON = 0.95
KS_LENGTH = 0
K_MAX = 6
DIR_NAME = "sphere_hough_"

SPHERES = [
    "rg",
    "rb",
    "gb",
]


def cos_of_vector(p1_bgr, p2_bgr):
    """
    p1, p2:(r,g,b) point
    used of calculation of rotation matrix
    """
    v1 = np.array((float(p1_bgr[0]), float(p1_bgr[1]), float(p1_bgr[2])))
    v2 = np.array((float(p2_bgr[0]), float(p2_bgr[1]), float(p2_bgr[2])))
    l1 = np.sqrt(v1.dot(v1))
    l2 = np.sqrt(v2.dot(v2))
    # ??? foget the function of this codes
    # if l1*l2 == 9:
    #     return 1
    return v1.dot(v2) / (l1 * l2)


class ColorRemover:
    def __init__(self, img_name):
        self.img_name, self.img_type = img_name.split(".")
        self.img = bm.Input(img_name)
        size = self.img.img.shape

        # A matrix with same size of input, to store the values after adjusted
        # For the reason that the adjusted values may over the range of (0, 255), the  vakye type here is float
        self.img_after_adjust = np.zeros(size, dtype=np.float)

        self.adjusted_img = self.img.img.copy()  # used for showing

        self.gray_after_adjust = self.img.gray.copy()
        self.slice_width = 0

        # For hough transform
        self.sphere_maps = {}
        self.hough_spheres = {}

        self.pixels_of_angle_in_sphere = {}
        self.points_of_angle_in_sphere = {}
        self.points_of_hough_line_in_sphere = {}
        self.hough_line_of_point_in_sphere = {}
        self.hough_lines_of_point = {}
        self.points_of_hough_line = {}
        self.pixels_of_hough_line = {}

        # For cluster based on hough transform
        self.labels_map = np.array([[-1]*size[1]]*size[0])
        self.cluster_bgr = {}
        self.pixels_of_hough_line_in_sphere = {}

        # For create color lines
        self.color_line_points = {}
        self.color_lines = {}

        # For cluster based on color line
        self.points_of_color_line = {}
        self.pixels_of_color_line = {}
        self.color_line_of_pixel = {}
        self.color_line_of_point = {}

        # For calculation of k,b
        self.pixel_edges = {}

        self.color_line_adjusted = {}
        self.pixels_in_slice = {}

        # For generating hemisphere and slices
        self.norm_map = self.generate_norm_map()
        norm_max = np.max(self.norm_map)
        norm_min = np.min(self.norm_map)
        width_slice = (norm_max - norm_min) / float(HEMISPHERE_NUM)
        self.slice_width = width_slice
        self.slice_start_norm = norm_min

    def generate_norm_map(self):
        bgr_flap = self.img.bgr.reshape(self.img.cols*self.img.rows, 3)
        norms = map(lambda bgr: np.sqrt(np.int32(bgr).dot(bgr)), bgr_flap)
        norm_map = np.array(norms).reshape(self.img.rows, self.img.cols)
        return norm_map

    def generate_color_lines(self):
        """
        read color line points of each cluster(label), then connect them as lines.
        If angle between neighbor lines is small enough then merge then as one single line.
        :return:
        """
        for label in self.color_line_points:
            points = self.color_line_points[label]
            line = tuple(points[:2])  # first two points as the first line
            for i in range(2, len(points)):
                next_line = line[1], points[i]
                # test angle. If the angle is small enough, then they are considered as one same line.
                vec = np.array(line[1]) - np.array(line[0])
                next_vec = np.array(next_line[1]) - np.array(next_line[0])
                cos = cos_of_vector(vec, next_vec)
                if cos > COS_FOR_SKELETON:  # close enough
                    line = line[0], next_line[1]
                else:
                    self.color_lines[label] = self.color_lines.get(label, []) + [line, ]
                    line = next_line
            self.color_lines[label] = self.color_lines.get(label, []) + [line, ]

    def cluster_points_to_color_line(self):
        """
        Attatch points of one cluster to the color lines belongs to this cluster.
        slice the ponits
        :return:
        """
        def cluster_points_label(label_of_point):
            pixels = self.pixels_of_hough_line[label_of_point]
            lines = self.color_lines[label_of_point]
            for pixel in pixels:
                point = self.img.get_bgr_value(pixel)
                norm = self.img.points_norms[point]
                if norm < self.img.get_bgr_norm(lines[0][0]):
                    self.points_of_color_line[lines[0]] = self.points_of_color_line.get(lines[0], []) + [point]
                    self.pixels_of_color_line[lines[0]] = self.pixels_of_color_line.get(lines[0], []) + [pixel]
                    self.color_line_of_pixel[pixel] = lines[0]
                    self.color_line_of_point[point] = lines[0]
                if norm > self.img.get_bgr_norm(lines[-1][1]):
                    self.points_of_color_line[lines[-1]] = self.points_of_color_line.get(lines[-1], []) + [point]
                    self.pixels_of_color_line[lines[-1]] = self.pixels_of_color_line.get(lines[-1], []) + [pixel]
                    self.color_line_of_pixel[pixel] = lines[-1]
                    self.color_line_of_point[point] = lines[-1]
                for line in lines:
                    if self.img.get_bgr_norm(line[0]) <= norm <= self.img.get_bgr_norm(line[1]):
                        self.points_of_color_line[line] = self.points_of_color_line.get(line, []) + [point]
                        self.pixels_of_color_line[line] = self.pixels_of_color_line.get(line, []) + [pixel]
                        self.color_line_of_pixel[pixel] = line
                        self.color_line_of_point[point] = line
                        break

        for label in self.points_of_hough_line:
            cluster_points_label(label)

    def calculate_edge_of_color_line(self, color_line):
        """
        This function generate Edges.
        Edge is in form of dict of dict with key of two color line: (p1, p2),(q1, q2) and values of list of pixels.
        If pixel1 and pixel2 are neighbors and belong to different color line then (pixel1, pixel2) are to one edge.
        Try to find pp, qq that 2 pixels away from edge. If they do not exist then use p, q that 1 pixel away from
        edge as edge pixes.
        :param color_line:
            Take color_line as the first color line of edge.
        :return:
            pixel_edges
        """
        direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # p, q are edge pixels one pixel away from edge;
        # pp, qq are pixels two pixels away from edge

        for (pixel_x, pixel_y) in self.pixels_of_color_line[color_line]:
            # 1. find p first
            p = (pixel_x, pixel_y)
            point_p = self.img.get_bgr_value(p)
            norm_p = self.img.points_norms[point_p]
            if norm_p == 0:
                continue
            for (x, y) in direction:
                # 2. find q
                q = (pixel_x + x, pixel_y + y)
                if q not in self.color_line_of_pixel:
                    continue
                neighbor_color_line = tuple(self.color_line_of_pixel[q])
                if neighbor_color_line == color_line:
                    continue
                pixel = p
                neighbor_pixel = q
                # p,q are edge pixels
                # qq = (pixel_x + x + x, pixel_y + y + y)
                # pp = (pixel_x - x, pixel_y - y)
                # if pp in self.pixels_of_color_line[color_line]:
                #     point_pp = self.img.get_bgr_value(pp)
                #     norm_pp = self.img.get_bgr_norm(point_pp)
                #     if norm_pp == 0:
                #         pp = None
                # else:
                #     pp = None
                #
                # if qq in self.pixels_of_color_line[neighbor_color_line]:
                #     point_qq = self.img.get_bgr_value(qq)
                #     norm_qq = self.img.get_bgr_norm(point_qq)
                #     if norm_qq == 0:
                #         qq = None
                # else:
                #     qq = None
                #
                # if pp and qq:
                #     pixel = pp
                #     neighbor_pixel = qq
                # else:
                #     pixel = p
                #     neighbor_pixel = q

                if color_line not in self.pixel_edges:
                    self.pixel_edges[color_line] = {neighbor_color_line: [(pixel, neighbor_pixel)]}
                elif neighbor_color_line not in self.pixel_edges[color_line]:
                    self.pixel_edges[color_line][neighbor_color_line] = [(pixel, neighbor_pixel)]
                else:
                    self.pixel_edges[color_line][neighbor_color_line].append((pixel, neighbor_pixel))

    def calculate_kb(self, color_line, to_color_line):
        """
        Calculate K for each edge of (color_line, to_color_line).
        Apply k to color_line to match color_line to to_color_line.
        :param color_line:
            First color line of edge
        :param to_color_line:
            Second color line of edge
        :return:
            Float of k
        """
        if color_line not in self.pixel_edges or to_color_line not in self.pixel_edges[color_line]:
            return 1
        edge_pixels = self.pixel_edges[color_line][to_color_line]
        # calculate the linear of the edge pixels
        points, to_points = [], []

        for pixel, to_pixel in edge_pixels:
            points += self.img.get_bgr_value(pixel)
            to_points += self.img.get_bgr_value(to_pixel, self.img_after_adjust)
            # points.append(self.img.get_bgr_value(pixel, self.img_after_adjust)[1])
            # to_points.append(self.img_after_adjust.item((x, y, 1)))

        # generate hough lines
        hough_points = zip(points, to_points)
        hough_convas = np.zeros((400, 400), dtype=np.uint8)
        hough_convas_3 = np.zeros((400, 400, 3), dtype=np.uint8)
        print hough_points
        for p in hough_points:
            x, y = map(int, p)
            if  0 < p[0] < 255 and 1 < p[1] < 255:
                hough_convas.itemset((x, y), 255)
                cv2.imshow("hough edge", hough_convas)
                cv2.waitKey()
        lines = cv2.HoughLinesP(hough_convas, 1, np.pi/180, HOUGH_VOTE, maxLineGap=100)[0]

        # calculate k and b
        ks = []
        bs = []
        for (x1, y1, x2, y2) in lines:
            if x1 == x2:
                continue
            color = [random.randint(10, 255), random.randint(100, 255), random.randint(1, 255)]
            cv2.line(hough_convas_3, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            ks.append((y2-y1)/float(x2-x1))
            bs.append(y1 - ks[-1]*x1)
        cv2.imshow("hough_lines", hough_convas_3)
        cv2.waitKey()
        k = sum(ks)/len(ks)
        b = sum(bs)/len(bs)
        return k, b

    @staticmethod
    def normalize(mat):
        """
        normalize 3D mat to 0~255
        if max - min < 255, then move it to the middle of 2~255
        else move it to 0~, then to 0~255
        :param mat:
            3D mat
        :return:
            mat
        """
        max_v = mat.max()
        min_v = mat.min()
        rng = max_v - min_v
        if rng <= 255:
            lower = min_v - (255 - rng) / 2
            mat -= lower
        else:
            # move to 0~
            mat = mat - min_v + 60
            k = 254.0 / (mat.max())
            print k
            mat *= k
        # print "mat_max, min:", mat.max(), mat.min()
        # print "mat:", mat
        return mat

    def transform_points(self, color_line, merge_to=None):
        """
        Transform points belont to color_line
        Generate Matrix for translate and rotate, caculate k.
        :param color_line:
            The points belongs to color_line will be transformed.
        :param merge_to, pixel_edges:
            If merge_to exists, then k will be calculated.
        :return:
            Nothing
        """
        # print "adjust ", color_line
        # translate point to matrix to make calculate simple
        matrix_p0 = np.float32(color_line[0])
        matrix_p1 = np.float32(color_line[1])
        # position of start point after transform
        norm_bgr_power = matrix_p0.dot(matrix_p0)
        norm_back = np.sqrt(norm_bgr_power / 3)
        # matrix of color line
        matrix_color_line = matrix_p1 - matrix_p0
        b_v, g_v, r_v = matrix_color_line
        b_p0, g_p0, r_p0 = matrix_p0
        # length of projection of line on bg
        norm_bg = math.sqrt(b_v ** 2 + g_v ** 2)

        # list for filter values after adjusted
        filter_list = []
        # angles for rotating
        alpha = math.pi / 2 if norm_bg == 0 else math.atan(r_v / float(norm_bg))
        beta = math.acos(b_v / float(norm_bg)) if norm_bg != 0 else 0
        theta = math.atan(math.sqrt(2) / 2) - alpha
        gama = math.pi / 4 - beta
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_beta_gama = math.cos(gama + beta)
        sin_beta_gama = math.sin(gama + beta)
        cos_beta = math.cos(beta)
        sin_beta = math.sin(beta)

        # rotate matrix
        matrix_z0 = np.matrix(
            [[cos_beta, sin_beta, 0],
             [-sin_beta, cos_beta, 0],
             [0, 0, 1],
             ]
        )

        matrix_y = np.matrix(
            [[cos_theta, 0, -sin_theta],
             [0, 1, 0],
             [sin_theta, 0, cos_theta],
             ]
        )

        matrix_z1 = np.matrix([
            [cos_beta_gama, -sin_beta_gama, 0],
            [sin_beta_gama, cos_beta_gama, 0],
            [0, 0, 1],
        ])

        # start to adjust
        k, b = 1, norm_back

        for pixel in self.pixels_of_color_line[color_line]:
            blue, green, red = self.img.get_bgr_value(pixel)
            print red, green, blue, "-",

            # move to origin
            red -= np.float(r_p0)
            green -= np.float(g_p0)
            blue -= np.float(b_p0)
            print red, green, blue, "-",
            # rotate to x=y=z
            blue, green, red = map(lambda a: a.item(0, 0), matrix_z0 * np.matrix([[blue], [green], [red]]))
            blue, green, red = map(lambda a: a.item(0, 0), matrix_y * np.matrix([[blue], [green], [red]]))
            blue, green, red = map(lambda a: a.item(0, 0), matrix_z1 * np.matrix([[blue], [green], [red]]))
            print red, green, blue
            x, y = pixel
            self.img_after_adjust.itemset((x, y, 0), blue)
            self.img_after_adjust.itemset((x, y, 1), green)
            self.img_after_adjust.itemset((x, y, 2), red)

        if merge_to:
            # adjust length of color line
            k, b = self.calculate_kb(color_line, merge_to)
        print "k, b:", k, b
        for pixel in self.pixels_of_color_line[color_line]:
            blue, green, red = map(lambda v: v*k+b, self.img.get_bgr_value(pixel, self.img_after_adjust))
            x, y = pixel
            print red, green, blue
            self.img_after_adjust.itemset((x, y, 0), blue)
            self.img_after_adjust.itemset((x, y, 1), green)
            self.img_after_adjust.itemset((x, y, 2), red)

        v_min = self.img_after_adjust.min()
        if v_min < 0:
            self.img_after_adjust -= v_min
        self.color_line_adjusted[color_line] = True
        _dic = self.pixel_edges[color_line]
        dic = sorted(_dic.iteritems(), key=lambda d: len(d[1]), reverse=True)
        dic = map(lambda d: d[0], dic)
        for line in dic:
            if line not in self.color_line_adjusted and line in self.pixels_of_color_line and line in self.pixel_edges:
                # if len(self.points_of_color_line[color_line]) < 50:
                #     continue
                self.transform_points(line, color_line)

    def adjust_color(self):
        """
        Main process of color adjustment based on color lines.
        :return:
            Nothing
        """
        print("generate edges...")
        for color_line in self.pixels_of_color_line:
            self.calculate_edge_of_color_line(color_line)
        print("start to adjust...")
        for color_line in self.pixels_of_color_line:
            if not self.pixels_of_color_line[color_line]:
                continue

            if color_line not in self.color_line_adjusted and color_line in self.pixel_edges:
                # if len(self.points_of_color_line[color_line]) < 50:
                #     continue
                self.transform_points(color_line)
        # normalize _img
        self.show_histogram_of_mat(self.img_after_adjust, "histogram of _img after adjust before normalize")
        self.img_after_adjust = self.normalize(self.img_after_adjust)
        self.show_histogram_of_mat(self.img_after_adjust, "histogram of _img after adjust")
        for x, y in self.img.pixels:
            for c in range(3):
                v = self.img_after_adjust.item((x, y, c))
                self.adjusted_img.itemset((x, y, c), v)
        for x, y in self.img.bg_pixels:
            for c in range(3):
                self.adjusted_img.itemset((x, y, c), 255)
        # cv2.imshow("before normalize", self.adjusted_img)
        # cv2.waitKey()
        return cv2.cvtColor(self.adjusted_img, cv2.COLOR_RGB2GRAY, self.gray_after_adjust)

    def show_lines_points(self, lines, points, title="lines & points"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title=title)
        self.img.fig_set_label(ax)
        self.img.fig_draw_lines(ax, lines)
        self.img.fig_draw_points(ax, points)
        # self.img.draw_hemi(ax, HEMISPHERE_NUM, self.slice_width, self.slice_start_norm)
        plt.show()

    def show_lines(self, lines, title="lines"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title=title)
        self.img.fig_set_label(ax)
        self.img.fig_draw_lines(ax, lines)
        # self.img.draw_hemi(ax, int(HEMISPHERE_NUM), self.slice_width)
        plt.show()

    def show_points(self, points, title="points"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title=title)
        self.img.fig_set_label(ax)
        points = set(map(lambda x: tuple(x), points))
        self.img.fig_draw_points(ax, points)
        # self.img.draw_hemi(ax, HEMISPHERE_NUM, self.slice_width)

    def show_points_2(self, points1, points2, title1="points1", title2="points2"):
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection="3d", title=title1)
        self.img.fig_set_label(ax1)
        points1 = set(map(lambda x: tuple(x), points1))
        self.img.fig_draw_points(ax1, points1)
        self.img.draw_hemi(ax1, HEMISPHERE_NUM, self.slice_width)

        ax2 = fig.add_subplot(122, projection="3d", title=title2)
        self.img.fig_set_label(ax2)
        points2 = set(map(lambda x: tuple(x), points2))
        self.img.fig_draw_points(ax2, points2)
        self.img.draw_hemi(ax2, HEMISPHERE_NUM, self.slice_width)
        plt.show()

    def show_bgr_histogram(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title="BGR Histogram")
        self.img.fig_set_label(ax)
        points = map(lambda x: tuple(x), self.img.points.keys())
        self.img.fig_draw_points(ax, points)
        # self.img.draw_hemi(ax, HEMISPHERE_NUM, self.slice_width, self.slice_start_norm)
        plt.show()

    def show_histogram_of_mat(self, mat, title="Mat Histogram", limit=300):
        mat_ = mat.reshape((1, mat.size/3, 3))
        points = map(lambda x: tuple(x), mat_[0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title=title)
        self.img.fig_set_label(ax, limit=limit)
        points = set(points)
        self.img.fig_draw_points(ax, points)
        plt.show()

    def show_label_points(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title="segments")
        self.img.fig_set_label(ax, limit=300)
        for label in self.cluster_bgr:
            points = set(self.cluster_bgr[label])
            cmap = [(np.random.random(3))]
            self.img.fig_draw_points(ax, points, cmap)
        plt.show()

    def show_label_points_1(self):
        for label in self.cluster_bgr:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d", title="segments")
            self.img.fig_set_label(ax, limit=300)
            points = set(self.cluster_bgr[label])
            if not points:
                pass
            cmap = [(np.random.random(3))]
            self.img.fig_draw_points(ax, points, cmap)
            plt.show()

    def show_cluster_points_1(self, title="points of line"):
        for line in self.points_of_color_line:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d", title=title)
            self.img.fig_set_label(ax)
            print line
            self.img.fig_draw_lines(ax, [line, ])
            self.img.fig_draw_points(ax, self.points_of_color_line[line])
            plt.show()

    def show_cluster_points_hough_1(self, title="points of hough line"):
        for line in self.points_of_hough_line_in_sphere:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d", title=title)
            self.img.fig_set_label(ax)
            print line
            # self._img.fig_draw_lines(ax, [line, ])
            self.img.fig_draw_points(ax, self.points_of_hough_line_in_sphere[line])
            plt.show()

    def show_pixels(self, pixels, num=0):
        img = np.zeros(self.img.bgr.shape, np.uint8)
        rows, cols, _ = img.shape
        for row, col in [(r, c) for r in range(rows) for c in range(cols)]:
            img.itemset((row, col, 0), 255)
            img.itemset((row, col, 1), 255)
            img.itemset((row, col, 2), 255)
        for x, y in pixels:
            img.itemset((x, y, 0), self.img.bgr.item((x, y, 0)))
            img.itemset((x, y, 1), self.img.bgr.item((x, y, 1)))
            img.itemset((x, y, 2), self.img.bgr.item((x, y, 2)))
        dir_name = DIR_NAME + self.img.img_name
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        fn = dir_name+"/colorregion"+str(num)+".bmp"
        cv2.imwrite(fn, img)

    @staticmethod
    def km_houg_lines(hough_lines):
        # hough_lines:(A, B, C, x1, y1, x2, y2)
        # k-means to cluster lines
        merged_lines = []
        descriptors = map(lambda line: line[0:2], hough_lines)
        descriptors = np.array(descriptors)
        # initial k-m
        km = KMeans(n_clusters=NUM_OF_HOUGH_LINE, max_iter=600)
        # apply k-m
        labels = km.fit_predict(descriptors)
        labels_set = set(labels)
        labels_arr = np.array(labels)
        hough_lines_arr = np.array(hough_lines)
        for label in labels_set:
            lines = hough_lines_arr[labels_arr == label]
            if len(lines) < 2:
                merged_lines.append(lines[0])
                continue
            # merged line
            max_len = 0
            max_line = lines[0]
            for line in lines:
                x1, y1, x2, y2 = line[-4:]
                length = np.power(y2-y1, 2)+np.power(x2-x1, 2)
                if length > max_len:
                    max_line = line
            merged_lines.append(max_line)
        return merged_lines

    def generate_hough_line(self):
        def cvt_polar(point_rgb):
            x, y, z = point_rgb
            p = np.array(np.float32(point_rgb))
            r = np.sqrt(p.dot(p))
            rr = np.sqrt(float(x)*x+float(y)*y)
            cosa = np.sqrt(2/3.0) if r == 0 else rr/float(r)
            cosb = np.sqrt(2)/2 if rr == 0 else x/float(rr)
            if cosa > 1:
                cosa = 1
            if cosb > 1:
                cosb = 1
            angles = np.degrees((np.arccos(cosa)*4, np.arccos(cosb)*4))
            return tuple(angles.astype(int))

        # project RGB points to sphere
        # 0. without moving O
        # 1. with moving O in direction of R
        # 2. with moving O in direction of G
        # 3. with moving O in direction of b

        for sphere in SPHERES:
            # init vars
            # pixels and points that is with a position on sphere
            # used for clustering
            # one point(angle of polar coordinate) on sphere corresponding to some pixels/points

            # result for clustering RGB points to hough lines.
            self.points_of_hough_line_in_sphere[sphere] = {}
            self.hough_line_of_point_in_sphere[sphere] = {}

            self.pixels_of_angle_in_sphere[sphere] = {}
            self.points_of_angle_in_sphere[sphere] = {}
            self.hough_spheres[sphere] = np.zeros((400, 400, 3), dtype=np.uint8)  # for show
            self.sphere_maps[sphere] = np.zeros((400, 400), dtype=np.uint8)  # for HT

            if sphere in ["r", "g", "b", "rgb"]:
                for pixel in self.img.fg_pixels:
                    point = self.img.get_bgr_value(pixel)
                    pb, pg, pr = point
                    # move axis
                    if "r" == sphere:
                        point = (pb, pg, pr+50)
                    if "g" == sphere:
                        point = (pb, pg+50, pr)
                    if "b" == sphere:
                        point = (pb+50, pg, pr)

                    # polar coordinate
                    a, b = cvt_polar(point)
                    self.pixels_of_angle_in_sphere[sphere][(a, b)] = \
                        self.pixels_of_angle_in_sphere[sphere].get((a, b), []) + [pixel]
                    self.points_of_angle_in_sphere[sphere][(a, b)] = \
                        self.points_of_angle_in_sphere[sphere].get((a, b), []) + [self.img.get_bgr_value(pixel)]
                    self.sphere_maps[sphere].itemset((b, a), 255)
                    self.hough_spheres[sphere].itemset((b, a, 0), 255)
                    self.hough_spheres[sphere].itemset((b, a, 1), 255)
                    self.hough_spheres[sphere].itemset((b, a, 2), 255)
            if sphere in ["rg", "rb", "gb"]:
                for pixel in self.img.fg_pixels:
                    point = self.img.get_bgr_value(pixel)
                    pb, pg, pr = point
                    # project points to plane
                    if "rg" == sphere:
                        a, b = (pr, pg)
                    if "gb" == sphere:
                        a, b = (pg, pb)
                    if "rb" == sphere:
                        a, b = (pr, pb)

                    self.pixels_of_angle_in_sphere[sphere][(a, b)] = \
                        self.pixels_of_angle_in_sphere[sphere].get((a, b), []) + [pixel]
                    self.points_of_angle_in_sphere[sphere][(a, b)] = \
                        self.points_of_angle_in_sphere[sphere].get((a, b), []) + [self.img.get_bgr_value(pixel)]
                    self.sphere_maps[sphere].itemset((b, a), 255)
                    self.hough_spheres[sphere].itemset((b, a, 0), 255)
                    self.hough_spheres[sphere].itemset((b, a, 1), 255)
                    self.hough_spheres[sphere].itemset((b, a, 2), 255)

            # do hough transform and generate hough lines
            hough_lines_sphere = []
            # cv2.imshow("hough_sphere_" + sphere, self.hough_spheres[sphere])
            # cv2.waitKey()
            sp_map = np.copy(self.hough_spheres[sphere])
            lines = cv2.HoughLinesP(self.sphere_maps[sphere], 1, np.pi/180, HOUGH_VOTE, maxLineGap=70)[0]
            i = 0
            for x1, y1, x2, y2 in lines:
                # line: Ax+By+C=0

                # a0 = y2-y1 if y2 != y1 else 0.0001
                # a = 1.0
                # b = (x1-x2)/float(a0)
                # c = (x2*y1-x1*y2)/(x2-x1)/float(a0)

                if y2 == y1:
                    a = 0.0
                    b = 1.0
                    c = float(-y1)
                elif x2 == x1:
                    a = 1.0
                    b = 0.0
                    c = float(-x1)
                else:
                    a0 = y2-y1
                    a = 1.0
                    b = (x1-x2)/float(a0)
                    c = (x2*y1-x1*y2)/(x2-x1)/float(a0)
                    # c = (x2-x1)*y1/float(y2-y1)-x1
                hough_lines_sphere.append((a, b, c, x1, y1, x2, y2))

            if len(hough_lines_sphere) > NUM_OF_HOUGH_LINE:
                hough_lines_sphere = self.km_houg_lines(hough_lines_sphere)

            for a, b, c, x1, y1, x2, y2 in hough_lines_sphere:
                color = [random.randint(10, 255), random.randint(100, 255), random.randint(1, 255)]
                # color[0] = 255
                # color[(i + 1) % 3] = 255
                i += 1
                color = tuple(color)
                cv2.line(sp_map, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                print "hough lines:", (a, b, c)

            # cv2.imshow("hough lines " + sphere, sp_map)
            # cv2.waitKey()

            # cluster
            min_lines = []
            for angle_a, angle_b in self.pixels_of_angle_in_sphere[sphere]:
                min_dist = 360*360*2
                min_line = hough_lines_sphere[0]
                for a, b, c, x1, y1, x2, y2 in hough_lines_sphere:
                    dist = np.power(a*angle_a+b*angle_b+c, 2)/(a*a+b*b)
                    if dist < min_dist:
                        min_dist = dist
                        min_line = (x1, y1, x2, y2)
                    if dist == min_dist:
                        xs = [x1, x2]
                        ys = [y1, y2]
                        xs.sort()
                        ys.sort()
                        if xs[0] < angle_a < xs[1] and ys[0] < angle_b < ys[1]:
                            min_dist = dist
                            min_line = (x1, y1, x2, y2)
                if sphere=="rb" and min_line not in min_lines:
                    min_lines.append(min_line)

                pixels = self.pixels_of_angle_in_sphere[sphere][(angle_a, angle_b)]
                self.points_of_hough_line_in_sphere[sphere][min_line] = \
                    self.points_of_hough_line_in_sphere[sphere].get(min_line, []) +\
                    [self.img.get_bgr_value(p) for p in pixels]
                for point in self.points_of_hough_line_in_sphere[sphere][min_line]:
                    self.hough_line_of_point_in_sphere[sphere][point] = min_line
            if sphere=="rb":
                print "mine_lines " + sphere, ":", min_lines
                print self.points_of_hough_line_in_sphere[sphere].keys()
                print "lines:",hough_lines_sphere
                rb_sphere = np.zeros((400, 400, 3), dtype=np.uint8)  # for show
                for line in min_lines:
                    cv2.line(rb_sphere, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [255, 255, 255])
                # cv2.imshow("min_lines", rb_sphere)
                # cv2.waitKey()
        # merge the results of cluster
        for pixel in self.img.fg_pixels:
            point = self.img.get_bgr_value(pixel)
            # lines = tuple(self.hough_line_of_point_in_sphere[sp][point] for sp in self.hough_line_of_point_in_sphere)
            lines = []
            for sp in self.hough_line_of_point_in_sphere:
                line = self.hough_line_of_point_in_sphere[sp][point]
                lines += list(line)
            lines = tuple(lines)
            self.hough_lines_of_point[point] = lines
            self.points_of_hough_line[lines] = self.points_of_hough_line.get(lines, ()) + (point,)
            self.pixels_of_hough_line[lines] = self.pixels_of_hough_line.get(lines, ()) + (pixel,)
        # cv2.waitKey()

    def meanshift_for_hough_line(self):
        # init mean shift
        pixels_of_label = {}
        points_of_label = {}
        for hough_line in self.points_of_hough_line:
            pixels = self.pixels_of_hough_line[hough_line]
            pixels = np.array(pixels)
            bandwidth = estimate_bandwidth(pixels, quantile=QUANTILE, n_samples=500)
            if bandwidth == 0:
                bandwidth = 2
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(pixels)
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            for k in range(n_clusters_):
                label = list(hough_line)
                label.append(k)
                pixels_of_label[tuple(label)] = map(tuple, pixels[labels==k])
        for label in pixels_of_label:
            pixels = pixels_of_label[label]
            points = map(self.img.get_bgr_value, pixels)
            points_of_label[label] = points
        self.pixels_of_hough_line = pixels_of_label
        self.points_of_hough_line = points_of_label

    def main_process_of_color_line_sphere_hough(self):
        print "Projecting pixels to sphere...",
        self.generate_hough_line()
        print "done"
        print "locally...",
        if LOCAL:
            self.meanshift_for_hough_line()
        print "locally done"
        # self.show_cluster_points_hough_1()
        # self.show_label_points_1()
        print "done"
        print "Generating color line points...",
        self.generate_color_points_with_label(self.points_of_hough_line)
        print "done"
        print "Generating color lines...",
        self.generate_color_lines()
        print "done"
        print "Clustering BGR points...",
        self.cluster_points_to_color_line()
        print "done"
        # self.show_cluster_points_1()
        num = 0
        for line in self.points_of_color_line:
            # points = self.line_points[line]
            # lines = [line, ]
            # self.show_lines_points(lines, points)
            pixels = self.pixels_of_color_line[line]
            self.show_pixels(pixels, num)
            num += 1

        # self.show_lines(self.line_pixels.keys(), "color_lines")
        self.show_lines(self.pixels_of_color_line.keys())
        # self.show_lines_points(self.pixels_of_color_line.keys(), self.color_line_of_point.keys())
        # for cl in points_cluster_to_line:
        #     self.show_lines_points([cl], points_cluster_to_line[cl], "clusters")

        print "Transfroming...",
        self.adjust_color()
        dir_name = DIR_NAME + self.img.img_name
        cv2.imwrite(dir_name+"/result.bmp", self.gray_after_adjust)
        # cv2.imshow("result", self.gray_after_adjust)
        # self.show_histogram_of_mat(self.gray_after_adjust, "result")
        # cv2.waitKey()
        # print self.calculate_difference()
        print "done"

    def generate_color_points_with_label(self, label_points_dict):
        """
        Use hemisphere to find intersect points with each cluster of BGR pints as color line points
        :return:
        """
        # calculate radios of hemispheres
        def is_intersect(point, r, thres=3):
            norm = self.img.points_norms[point]
            if np.abs(norm - r) > thres:
                return False
            return True

        def get_color_line_points(arr_points, slice_width, slice_start_norm):
            # unique
            points = set(map(tuple, arr_points))
            if not points:
                return None
            color_line_points = []
            # find intersect points
            for slice_seq in range(1, HEMISPHERE_NUM):
                r = slice_width*slice_seq + slice_start_norm
                intersect_points = filter(lambda point: is_intersect(point, r), points)
                if intersect_points:
                    color_line_points.append(tuple(map(np.mean, zip(*intersect_points))))
            # if points' number is less than 2 (not enough for skeleton) then
            # take the points with max and min norm value as color line points
            if len(color_line_points) < 2:
                # get norm values
                norm_point = map(lambda x: (self.img.points_norms[x], x), points)
                norms, points = zip(*norm_point)
                indic_max = np.argmax(norms)
                indic_min = np.argmin(norms)
                color_line_points = [points[indic_min],]+color_line_points+[points[indic_max],]
            return color_line_points

        for label in label_points_dict:
            print "LABEL:", label
            points = label_points_dict[label]
            norms = map(self.img.get_bgr_norm, points)
            start_norm = min(norms)
            end_norm = max(norms)
            slice_width = (end_norm-start_norm)/HEMISPHERE_NUM
            if points:
                self.color_line_points[label] = get_color_line_points(points,slice_width, start_norm)

    def calculate_difference(self):
        input_name = self.img_name
        img_type = self.img_type
        truth_name = input_name+"1."+img_type
        truth_img = bm.Input(truth_name)

        gray = self.img.gray
        gray_result = self.gray_after_adjust
        gray_truth = cv2.cvtColor(truth_img.img, cv2.COLOR_BGR2GRAY)

        resized_gray = self.img.resize_fg_gray(gray)
        resized_truth = truth_img.resize_fg_gray(gray_truth)
        resized_result = self.img.resize_fg_gray(gray_result)

        cv2.imshow("resized_gray", resized_gray)
        cv2.imshow("resized_truth", resized_truth)
        cv2.imshow("resized_result", resized_result)
        cv2.waitKey()

        diff_result = resized_result.astype(float) - resized_truth.astype(float)
        diff_result = diff_result[resized_result < 255]
        diff_gray = resized_gray.astype(float) - resized_truth.astype(float)
        diff_gray = diff_gray[resized_gray < 255]

        data = np.array([diff_result.flatten(), diff_gray.flatten()]).transpose()

        hist_truth = resized_truth[resized_truth<255]
        hist_result = resized_result[resized_result<255]
        hist_gray = resized_gray[resized_gray<255]

        plt.figure()
        plt.xlim(0, 255)
        n, bins, patches = plt.hist(hist_truth, 40, normed=1, alpha=0.75)
        plt.title('Histogram of truth')
        plt.xlabel('value')
        plt.ylabel('percent')
        plt.show()

        plt.figure()
        plt.xlim(0, 255)
        n, bins, patches = plt.hist(hist_result, 40, normed=1, alpha=0.75)
        plt.title('Histogram of result')
        plt.xlabel('value')
        plt.ylabel('percent')
        plt.show()

        plt.figure()
        plt.xlim(0, 255)
        n, bins, patches = plt.hist(hist_gray, 40, normed=1, alpha=0.75)
        plt.title('Histogram of gray')
        plt.xlabel('value')
        plt.ylabel('percent')
        plt.show()

        plt.figure()
        plt.xlim(-255, 255)
        n, bins, patches = plt.hist(diff_result, 40, normed=1, alpha=0.75)
        plt.title('Histogram of difference between result and truth')
        plt.xlabel('value')
        plt.ylabel('percent')
        plt.show()

        plt.figure()
        plt.xlim(-255, 255)
        n, bins, patches = plt.hist(diff_gray, 40, normed=1, alpha=0.75)
        plt.title('Histogram of difference between linear luminance and truth')
        plt.xlabel('value')
        plt.ylabel('percent')
        plt.show()

        # cv2.waitKey()
        return np.abs(diff_result).mean(), diff_result.std(), np.abs(diff_gray).mean(), diff_gray.std()

if __name__ == "__main__":
    files = [
        # "_img/man.bmp",
        # "_img/bird.bmp",
        # "_img/nasu.bmp",
        # "_img/teapot.png",
        # "_img/teapot0.png",
        # "_img/sp5080.png",
        # "_img/sp6.png",
        # "img/cylinder.png",
        IMAGE,#"local/tiger.png",
    ]

    # files = ["_img/bird.bmp"]
    for file in files:
        cr = ColorRemover(file)
        # cr.cluster_pixels()
        cr.show_bgr_histogram()
        cr.main_process_of_color_line_sphere_hough()
        # cv2.imshow("result gray r", cr.gray_after_adjust)
        # cv2.imshow("gray", cr._img.gray)
        # cv2.waitKey()
