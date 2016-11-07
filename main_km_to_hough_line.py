__author__ = 'wtq'
# apply KM when generate hough lines to reduce number of lines
# including 3 planes and 4 spheres
# coding=utf-8
__author__ = 'wtq'
import cv2
import basicimage as bm
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import os
from sklearn.decomposition import PCA

NUM_OF_HOUGH_LINE = 6
VECTOR_DIMENSION = 36
K = 20
HEMISPHERE_NUM = 30
COS_FOR_SKELETON = 0.95
KS_LENGTH = 0
K_MAX = 6
DIR_NAME = "sphere_hough_"

SPHERES = [
    # "rgb",
    # "r", "g",
    # "b",
    "rg",
    "rb",
    "gb",
]

def cos_of_vector(p1_bgr, p2_bgr):
    """
    p1, p2:(r,g,b) point
    """
    x, y, z = float(p1_bgr[0]), float(p1_bgr[1]), float(p1_bgr[2])
    v1 = np.array((x, y, z))
    x, y, z = float(p2_bgr[0]), float(p2_bgr[1]), float(p2_bgr[2])
    v2 = np.array((x, y, z))
    # (b2, g2, r2) = p2_bgr
    l1 = np.sqrt(v1.dot(v1))
    l2 = np.sqrt(v2.dot(v2))
    if l1*l2 == 9:
        return 1
    return v1.dot(v2) / (l1 * l2)

class ColorRemover:
    descriptor_map = None

    def __init__(self, img_name):
        self.img = bm.Input(img_name)
        size = self.img.img.shape
        self.img_after_adjust = np.zeros(size, dtype=np.int64)
        self.adjusted_img = self.img.img.copy()  # used for showing
        self.gray_after_adjust = self.img.gray.copy()
        self.slice_width = 0

        self.descriptor_map = None  # self.generate_descriptor_map()

        self.sphere_maps = {}
        self.hough_spheres = {}

        self.pixels_of_angle_in_sphere = {}
        self.points_of_angle_in_sphere = {}
        self.points_of_hough_line_in_sphere = {}
        self.hough_line_of_point_in_sphere = {}
        self.hough_lines_of_point = {}
        self.points_of_hough_line = {}
        self.pixels_of_hough_line = {}

        self.labels_map = np.array([[-1]*size[1]]*size[0])
        self.cluster_bgr = {}
        self.pixels_of_hough_line_in_sphere = {}
        self.color_line_points = {}
        self.color_lines = {}

        self.points_of_color_line = {}
        self.pixels_of_color_line = {}
        self.color_line_of_pixel = {}
        self.color_line_of_point = {}

        self.pixel_edges = {}
        self.color_line_adjusted = {}

        self.pixels_in_slice = {}

        self.norm_map = self.generate_norm_map()
        norm_max = np.max(self.norm_map)
        width_slice = norm_max / float(HEMISPHERE_NUM)
        self.slice_width = width_slice
        pass

    def slice_pixels(self):
        for pixel in self.img.fg_pixels:
            norm = self.img.get_bgr_norm(self.img.get_bgr_value(pixel))
            for i in range(HEMISPHERE_NUM):
                if norm < self.slice_width*(i+1):
                    self.pixels_in_slice[i] = self.pixels_in_slice.get(i, []) + [pixel]
                    break

    def generate_norm_map(self):
        bgr_flap = self.img.bgr.reshape(self.img.cols*self.img.rows, 3)
        norms = map(lambda bgr: np.sqrt(np.int32(bgr).dot(bgr)), bgr_flap)
        norm_map = np.array(norms).reshape(self.img.rows, self.img.cols)
        return norm_map

    @staticmethod
    def trans2spaces(img):
        print "    Transforming to spaces...",
        def trans_cgst(bgr):
            g = min(bgr)
            bgr_g = (bgr-g)
            bgr_g.sort()
            mm, m, _ = bgr_g
            s = m
            t = mm - m
            gst = np.array([g, s, t])
            gst.shape = (1, 3)
            return gst

        def trans_divbgr(bgr):
            b, g, r = bgr
            bg = float(b)/g
            gr = float(g)/r
            rb = float(r)/b
            return np.array([bg, gr, rb])

        raws, cols, _ = img.shape
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        yiq = np.ndarray((raws, cols, 3), dtype=np.float32)
        yiq_matrix = np.matrix(
            [[0.299, 0.587, 0.144],
             [0.596, -0.275, -0.321],
             [0.212, -0.528, 0.311]]
        )
        for (r, c) in [(raw, col) for raw in range(raws) for col in range(cols)]:
            bgr_array = img[r][c]
            bgr_array.shape = (1, 3)
            bgr_array = np.matrix(bgr_array)
            yiq_rc = bgr_array * yiq_matrix
            yiq[r][c] = yiq_rc

        gst = np.ndarray((raws, cols, 3), dtype=np.float32)
        for (r, c) in [(raw, col) for raw in range(raws) for col in range(cols)]:
            gst[r][c] = trans_cgst(img[r][c])

        divbgr = np.ndarray((raws, cols, 3), dtype=np.float32)
        for (r, c) in [(raw, col) for raw in range(raws) for col in range(cols)]:
            divbgr[r][c] = trans_divbgr(img[r][c])
        print "done"
        return {"ycrcb": ycrcb, "yuv": yuv, "hls": hls, "yiq": yiq, "bgr": img, "divbgr": divbgr, "gst": gst}

    def generate_descriptor_map(self):
        def cvt_polar(point):
            x, y, z = point
            p = np.array(np.float32(point))
            r = np.sqrt(p.dot(p))
            rr = np.sqrt(float(x)*x+float(y)*y)
            cosa = np.sqrt(2/3.0) if r == 0 else rr/float(r)
            cosb = np.sqrt(2)/2 if rr == 0 else x/float(rr)
            if cosa > 1:
                cosa = 1
            if cosb > 1:
                cosb = 1
            return np.arccos(cosa)/np.pi*510, np.arccos(cosb)/np.pi*510

        def cvt_projection(s_coor, r=442.0):
            a, b = s_coor
            a = a / 510.0 * np.pi
            b = b / 510.0 * np.pi
            x = r*np.cos(a)*np.cos(b)
            y = r*np.cos(a)*np.sin(b)
            z = r*np.sin(a)
            return x, y, z

        def func_for_map(pos):
            r, c = pos
            d_xy = (r/float(rows)*255, c/float(cols)*255)
            d_bgr = tuple(spaces["bgr"][r][c])
            s_bgr = cvt_polar(d_bgr)
            p_bgr = cvt_projection(s_bgr)

            d_hls = tuple(spaces["hls"][r][c])
            s_hls = cvt_polar(d_hls)
            p_hls = cvt_projection(s_hls)

            d_ycrcb = tuple(spaces["ycrcb"][r][c])
            s_ycrcb = cvt_polar(tuple(spaces["ycrcb"][r][c] - [0, 0, 128]))
            p_ycrcb = cvt_projection(s_ycrcb)

            d_yuv = tuple(spaces["yuv"][r][c])
            s_yuv = cvt_polar(tuple(spaces["yuv"][r][c] - [0, 0, 128]))
            p_yuv = cvt_projection(s_yuv)

            d_yiq = tuple(spaces["yiq"][r][c])
            s_yiq = cvt_polar(d_yiq)
            p_yiq = cvt_projection(s_yiq)
            # d_divbgr = tuple(spaces["divbgr"][r][c]/255.0)
            # s_divbgr = cvt_polar(d_divbgr)
            d_gst = tuple(spaces["gst"][r][c])
            s_gst = cvt_polar(d_gst)
            p_gst = cvt_projection(s_gst)
            h = (2*d_hls[0],)
            sdg = (d_gst[1]/float(d_gst[0]),) if d_gst[0] !=0 else (0,)
            vecter = np.array(()
                +s_bgr
                +sdg
                +s_yuv
                # +pol_ycrcb
                +h
                + d_xy
                # +d_bgr
                # +d_hls
                # +d_yiq
                # +d_gst
                +p_bgr
                # +p_hls
                # +p_yiq
                # +pro_gst
            )
            descriptor_map[r][c] = vecter
            return

        rows, cols, _ = self.img.img.shape
        descriptor_map = np.ndarray((rows, cols, VECTOR_DIMENSION), dtype=np.float32)
        spaces = self.trans2spaces(self.img.bgr)

        # pool = ThreadPool(6)
        # pool.map(func_for_map, [(row, col) for row in range(rows) for col in range(cols)])
        # pool.close()
        # pool.join()
        print "creating vectors..."
        # for (r, c) in [(row, col) for row in range(rows) for col in range(cols)]:
        for (r, c) in self.img.fg_pixels:
            xy = (r/float(rows)*255, c/float(cols)*255)

            bgr = tuple(spaces["bgr"][r][c])
            pol_bgr = cvt_polar(bgr)
            pro_bgr = cvt_projection(pol_bgr)

            hls = tuple(spaces["hls"][r][c])

            ycrcb = tuple(spaces["ycrcb"][r][c])
            pol_ycrcb = cvt_polar(tuple(spaces["ycrcb"][r][c] - [0, 0, 128]))
            pro_ycrcb = cvt_projection(pol_ycrcb)
            crcb = (ycrcb[1], ycrcb[2])
            # print "val:", ycrcb
            # print "pol:", pol_ycrcb
            # print "pro:", pro_ycrcb

            yuv = tuple(spaces["yuv"][r][c])
            pol_yuv = cvt_polar(tuple(spaces["yuv"][r][c] - [0, 0, 128]))
            uv = (yuv[1], yuv[2])
            pro_yuv = cvt_projection(pol_yuv)

            yiq = tuple(spaces["yiq"][r][c])
            pol_yiq = cvt_polar(yiq)
            pro_yiq = cvt_projection(pol_yiq)
            iq = (yiq[1], yiq[2])
            # d_divbgr = tuple(spaces["divbgr"][r][c]/255.0)
            # s_divbgr = cvt_polar(d_divbgr)
            gst = tuple(spaces["gst"][r][c])
            pol_gst = cvt_polar(gst)
            pro_gst = cvt_projection(pol_gst)
            h = (hls[0]*2,)
            sdg = gst[1]/float(gst[0]) if gst[0] != 0 else 0
            sdg = (np.arctan(sdg)/np.pi*510,)
            vector_group_1 = (pol_yiq+pro_bgr+pro_yiq+xy)
            vector_group_2 = (vector_group_1+yiq+yuv)
            vector_group_3 = (crcb+gst+hls+iq+pol_bgr+pol_ycrcb+pol_yiq+pol_yuv+pro_bgr+pro_gst+pro_ycrcb+pro_yiq+
                              pro_yuv+sdg+uv+xy)
            vector_group_4 = (crcb+gst+hls+iq+pol_bgr+pol_ycrcb+pol_yiq+pol_yuv+pro_bgr+pro_gst+pro_ycrcb+pro_yiq+
                              pro_yuv+sdg+uv)
            vector_group_5 = (crcb+h+iq+pol_yiq+pro_bgr+pro_yiq+uv+xy)
            vector_group_6 = (crcb+h+iq+pol_yiq+pro_bgr+pro_yiq+uv)
            vector = vector_group_4
            '''
            # polar coordinate
            # +pol_bgr
            # +sdg
            # +pol_yuv
            # +pol_ycrcb
            +pol_yiq

            # h channel
            +h

            # position in spaces
            # +xy
            # +bgr
            # +hls
            # +gst
            # +yiq
            # +ycrcb
            # +yuv
            + iq
            + crcb
            + uv

            # projection of points to hemisphere
            # +pro_ycrcb
            # +pro_yuv
            +pro_bgr
            +pro_yiq
            # +pro_gst
            '''

            # descriptor_map[r][c] = np.array(map(lambda x: x if x == x else 0, vecter))
            descriptor_map[r][c] = np.array(vector)
        self.descriptor_map = descriptor_map

    def cluster_pixels_km(self):
        """
        :type self: ColorRemover
        """
        fg_pixels = self.img.fg_pixels.keys()
        descriptors = []
        for r, c in fg_pixels:
            descriptors.append(self.descriptor_map[r][c])
        descriptors = np.array(descriptors)
        # print "PCA..."
        # descriptors = PCA(n_components=int(VECTOR_DIMENSION)/2).fit_transform(descriptors)
        # print "done"
        # initial k-m
        km = KMeans(n_clusters=K, max_iter=600)

        # apply k-m
        labels = km.fit_predict(descriptors)
        # ret,labels,center=cv2.kmeans(descriptors,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # self.labels_map = labels.reshape(self._img.rows, self._img.cols)

        for i in range(len(labels)):
            xy = fg_pixels[i]
            label = labels[i]
            self.labels_map.itemset(xy, label)
        # save the indices and BGR values of each cluster as a dictionary with keys of label
        for label in range(K):
            self.pixels_of_hough_line_in_sphere[label] = map(tuple, np.argwhere((self.labels_map == label)))
            self.cluster_bgr[label] = map(tuple, self.img.bgr[self.labels_map == label])

    def cluster_pixels(self):
        # reshape
        """
        :type self: ColorRemover
        """
        fg_pixels = self.img.fg_pixels.keys()
        descriptors = []
        for r, c in fg_pixels:
            descriptors.append(self.descriptor_map[r][c])
        descriptors = np.array(descriptors)
        # descriptors = self.descriptor_map.reshape(descriptors_rows, 1, VECTOR_DIMENSION)

        # initial k-m
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 3)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # apply k-m
        compactness, labels, centers = cv2.kmeans(descriptors, 10, criteria, 10, flags)
        # ret,labels,center=cv2.kmeans(descriptors,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # self.labels_map = labels.reshape(self._img.rows, self._img.cols)

        for i in range(len(labels)):
            xy = fg_pixels[i]
            label = labels[i]
            self.labels_map.itemset(xy, label)
        # save the indices and BGR values of each cluster as a dictionary with keys of label
        for label in range(K):
            self.pixels_of_hough_line_in_sphere[label] = map(tuple, np.argwhere((self.labels_map == label)))
            self.cluster_bgr[label] = map(tuple, self.img.bgr[self.labels_map == label])

    def cluster_pixels_ms(self):
        # reshape
        """
        cluster points descriptors by meahs shift
        :type self: ColorRemover
        """
        fg_pixels = self.img.fg_pixels.keys()
        descriptors = []
        for r, c in fg_pixels:
            descriptors.append(self.descriptor_map[r][c])
        descriptors = np.array(descriptors)
        descriptors = PCA(n_components=int(VECTOR_DIMENSION)/2).fit_transform(descriptors)
        # descriptors = self.descriptor_map.reshape(descriptors_rows, 1, VECTOR_DIMENSION)
        bandwidth = estimate_bandwidth(descriptors, quantile=0.05)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(descriptors)
        labels = ms.labels_

        for i in range(len(labels)):
            xy = fg_pixels[i]
            label = labels[i]
            self.labels_map.itemset(xy, label)
        # save the indices and BGR values of each cluster as a dictionary with keys of label
        for label in range(K):
            self.pixels_of_hough_line_in_sphere[label] = map(tuple, np.argwhere((self.labels_map == label)))
            self.cluster_bgr[label] = map(tuple, self.img.bgr[self.labels_map == label])

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

        def get_color_line_points(arr_points):
            # unique
            points = set(map(tuple, arr_points))
            if not points:
                return None
            color_line_points = []
            # find intersect points
            for slice_seq in range(1, HEMISPHERE_NUM+1):
                r = self.slice_width*slice_seq
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
            points = label_points_dict[label]
            if points:
                self.color_line_points[label] = get_color_line_points(points)

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
                # test angle
                vec = np.array(line[1]) - np.array(line[0])
                next_vec = np.array(next_line[1]) - np.array(next_line[0])
                cos = cos_of_vector(vec, next_vec)
                if cos > COS_FOR_SKELETON:  # close enough
                    line = line[0], next_line[1]
                else:
                    self.color_lines[label] = self.color_lines.get(label, []) + [line, ]
                    line = next_line
            self.color_lines[label] = self.color_lines.get(label, []) + [line, ]
            # if label in self.color_lines:
            #     self.color_lines[label].append(line)
            # else:
            #     self.color_lines[label] = [line, ]

    def cluster_points_to_color_line(self):
        """
        attatch points of one cluster to the color lines belongs to this cluster.
        slice the ponits
        :return:
        """
        def cluster_points_label(label):
            pixels = self.pixels_of_hough_line[label]
            lines = self.color_lines[label]
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
                for line in self.color_lines[label]:
                    if self.img.get_bgr_norm(line[0]) < norm < self.img.get_bgr_norm(line[1]):
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
        :param pixel_edges:
            The result of edges.
        :param color_line:
            Take color_line as the first color line of edge.
        :param pixels_belong_to_color_line:
            The dict with keys of color_line and values of pixels
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
            norm_p = self.img.get_bgr_norm(point_p)
            if norm_p == 0:
                continue
            for (x, y) in direction:
                # 2. find q
                q = (pixel_x + x, pixel_y + y)
                if q not in self.color_line_of_pixel:
                    continue
                neighbor_color_line = self.color_line_of_pixel[q]
                neighbor_color_line = tuple(neighbor_color_line)
                if neighbor_color_line == color_line:
                    continue
                # p,q are edge pixels
                qq = (pixel_x + x + x, pixel_y + y + y)
                pp = (pixel_x - x, pixel_y - y)
                if pp in self.pixels_of_color_line[color_line]:
                    point_pp = self.img.get_bgr_value(pp)
                    norm_pp = self.img.get_bgr_norm(point_pp)
                    if norm_pp == 0:
                        pp = None
                else:
                    pp = None

                if qq in self.pixels_of_color_line[neighbor_color_line]:
                    point_qq = self.img.get_bgr_value(qq)
                    norm_qq = self.img.get_bgr_norm(point_qq)
                    if norm_qq == 0:
                        qq = None
                else:
                    qq = None

                if pp and qq:
                    pixel = pp
                    neighbor_pixel = qq
                else:
                    pixel = p
                    neighbor_pixel = q

                if color_line not in self.pixel_edges:
                    self.pixel_edges[color_line] = {neighbor_color_line: [(pixel, neighbor_pixel)]}
                elif neighbor_color_line not in self.pixel_edges[color_line]:
                    self.pixel_edges[color_line][neighbor_color_line] = [(pixel, neighbor_pixel)]
                else:
                    self.pixel_edges[color_line][neighbor_color_line].append((pixel, neighbor_pixel))

    def calculate_k(self, color_line, to_color_line):
        """
        Calculate K for each edge of (color_line, to_color_line).
        Apply k to color_line to match color_line to to_color_line.
        :param color_line:
            First color line of edge
        :param to_color_line:
            Second color line of edge
        :param pixel_edges:
            dict of edge
        :return:
            Float of k
        """
        if color_line not in self.pixel_edges or to_color_line not in self.pixel_edges[color_line]:
            return 1
        edge_pixels = self.pixel_edges[color_line][to_color_line]
        # calculate difference of BGR norm between two segment around the edge
        ks = []
        nb = [-1, 1]
        for pixel, to_pixel in edge_pixels:
            points = [self.img.get_bgr_value(pixel), ]
            x, y = pixel
            for dx in nb:
                for dy in nb:
                    xx = x + dx
                    yy = y + dy
                    if (xx, yy) not in self.pixels_of_color_line[color_line]:
                        continue
                    points.append(self.img.get_bgr_value((xx, yy)))
            point_norm_list = map(self.img.get_bgr_norm, points)
            point_norm = sum(point_norm_list) / float(len(point_norm_list))
            if point_norm == 0:
                continue
            to_points = [self.img.get_bgr_value(to_pixel, self.img_after_adjust), ]
            x, y = to_pixel
            for dx in nb:
                for dy in nb:
                    xx = x + dx
                    yy = y + dy
                    if (xx, yy) not in self.pixels_of_color_line[to_color_line]:
                        continue
                    to_points.append(self.img.get_bgr_value((xx, yy), self.img_after_adjust))
            to_point_norm_list = map(self.img.get_bgr_norm, to_points)
            to_point_norm = sum(to_point_norm_list) / float(len(to_point_norm_list))
            _k = to_point_norm / float(point_norm)
            # if _k > 10:
            #     print "to/from list:", to_point_norm_list, point_norm_list
            #     print "to/from:", to_point_norm, point_norm
            ks.append(_k)

        # find best ks with least var
        ks.sort()
        mean = 1
        if len(ks) > KS_LENGTH:
            mean = np.mean(ks)
        if mean > K_MAX:
            mean = K_MAX
        print "ks:", ks
        return mean

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
        :param pixels_belong_to_color_line:
            Pixels to be transform
        :param merge_to, pixel_edges:
            If merge_to exists, then k will be calculated.
        :return:
            Nothing
        """
        print "adjust ", color_line
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
        # parameter K
        k = 1
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

        if merge_to:
            # adjust length of color line
            k = self.calculate_k(color_line, merge_to)
            print "KKKKKKKKKKKKK:", k
        for pixel in self.pixels_of_color_line[color_line]:
            blue, green, red = self.img.get_bgr_value(pixel)
            # move to origin
            red -= float(r_p0)
            green -= float(g_p0)
            blue -= float(b_p0)

            # rotate to x=y=z
            blue, green, red = map(lambda a: a.item(0, 0), matrix_z0 * np.matrix([[blue], [green], [red]]))
            blue, green, red = map(lambda a: a.item(0, 0), matrix_y * np.matrix([[blue], [green], [red]]))
            blue, green, red = map(lambda a: a.item(0, 0), matrix_z1 * np.matrix([[blue], [green], [red]]))

            # move to start point of segment
            blue, green, red = map(lambda a: a + norm_back, [blue, green, red])
            if merge_to:
                blue, green, red = map(lambda a: a * k, [blue, green, red])
            filter_list += [blue, green, red]

            x, y = pixel
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
                if len(self.points_of_color_line[color_line]) < 100:
                    continue
                self.transform_points(line, color_line)

    def adjust_color(self):
        """
        Main process of color adjustment based on color lines.
        :param pixels_belong_to_color_line:
        :param color_line_of_pixel:
            Dict with keys of pixels and values of color line the key belongs to
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
                if len(self.points_of_color_line[color_line]) < 100:
                    continue
                self.transform_points(color_line)
        # normalize _img
        self.show_histogram_of_mat(self.img_after_adjust, "histogram of _img after adjust before normalize", limit=7700)
        self.img_after_adjust = self.normalize(self.img_after_adjust)
        self.show_histogram_of_mat(self.img_after_adjust, "histogram of _img after adjust")
        for x, y in self.img.pixels_points:
            for c in range(3):
                v = self.img_after_adjust.item((x, y, c))
                self.adjusted_img.itemset((x, y, c), v)
        for x, y in self.img.bg_pixels:
            for c in range(3):
                self.adjusted_img.itemset((x, y, c), 255)
        return cv2.cvtColor(self.adjusted_img, cv2.COLOR_RGB2GRAY, self.gray_after_adjust)

    def show_lines_points(self, lines, points, title="lines & points"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title=title)
        self.img.fig_set_label(ax)
        self.img.fig_draw_lines(ax, lines)
        self.img.fig_draw_points(ax, points)
        self.img.draw_hemi(ax, HEMISPHERE_NUM, self.slice_width)
        plt.show()

    def show_lines(self, lines, title="lines"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title=title)
        self.img.fig_set_label(ax)
        self.img.fig_draw_lines(ax, lines)
        self.img.draw_hemi(ax, int(HEMISPHERE_NUM), self.slice_width)
        plt.show()

    def show_points(self, points, title="points"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", title=title)
        self.img.fig_set_label(ax)
        points = set(map(lambda x: tuple(x), points))
        self.img.fig_draw_points(ax, points)
        self.img.draw_hemi(ax, HEMISPHERE_NUM, self.slice_width)

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
        points = map(lambda x: tuple(x), self.img.pixels_points.values())
        points = set(points)
        self.img.fig_draw_points(ax, points)
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
        for x, y in pixels:
            img.itemset((x, y, 0),self.img.bgr.item((x, y, 0)))
            img.itemset((x, y, 1),self.img.bgr.item((x, y, 1)))
            img.itemset((x, y, 2),self.img.bgr.item((x, y, 2)))
        dir_name =DIR_NAME+self.img.img_name
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        fn = dir_name+"/colorregion"+str(num)+".bmp"
        cv2.imwrite(fn, img)

    def km_houg_lines(self, hough_lines):
        # hough_lines:(A, B, C, x1, y1, x2, y2)
        # k-means to cluster lines
        merged_lines = []
        descriptors = map(lambda line: line[1:3], hough_lines)
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
        def cvt_polar(point):
            x, y, z = point
            p = np.array(np.float32(point))
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
        # 3. with moving O in direction of B

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
            self.hough_spheres[sphere] = np.zeros((400, 400, 3), dtype=np.uint8) # for show
            self.sphere_maps[sphere] = np.zeros((400, 400), dtype=np.uint8) # for HT

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
            cv2.imshow("hough_sphere_" + sphere, self.hough_spheres[sphere])
            cv2.waitKey()
            sp_map = np.copy(self.hough_spheres[sphere])
            lines = cv2.HoughLinesP(self.sphere_maps[sphere], 1, np.pi/180, 60, maxLineGap=50)[0]
            i = 0
            for x1, y1, x2, y2 in lines:
                # line: Ax+By+C=0
                A0 = y2-y1 if y2 != y1 else 0.0001
                A = 1.0
                B = (x1-x2)/float(A0)
                C = (x2*y1-x1*y2)/(x2-x1)/float(A0)
                hough_lines_sphere.append((A, B, C, x1, y1, x2, y2))

            if len(hough_lines_sphere) > NUM_OF_HOUGH_LINE:
                hough_lines_sphere = self.km_houg_lines(hough_lines_sphere)

            for a, b, c, x1, y1, x2, y2 in hough_lines_sphere:
                color = [0, 0, 0]
                color[i % 3] = 255
                i += 1
                color = tuple(color)
                cv2.line(sp_map, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            cv2.imshow("hough lines", sp_map)
            cv2.waitKey()

            # cluster
            for angle_a, angle_b in self.pixels_of_angle_in_sphere[sphere]:
                min_dist = 360*360*2
                min_line = hough_lines_sphere[0]
                for A, B, C, x1, y1, x2, y2 in hough_lines_sphere:
                    dist = np.power(A*angle_a+B*angle_b+C, 2)/(A*A+B*B)
                    if dist <= min_dist:
                        min_dist = dist
                        min_line = (B, C)

                piexls = self.pixels_of_angle_in_sphere[sphere][(angle_a, angle_b)]
                self.points_of_hough_line_in_sphere[sphere][min_line] = \
                    self.points_of_hough_line_in_sphere[sphere].get(min_line, []) +\
                    [self.img.get_bgr_value(p) for p in piexls]
                for point in self.points_of_hough_line_in_sphere[sphere][min_line]:
                    self.hough_line_of_point_in_sphere[sphere][point] = min_line

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

    def main_process_of_color_line_sphere_hough(self):
        print "Projecting pixels to sphere...",
        self.generate_hough_line()
        print "done"

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

        self.show_lines_points(self.pixels_of_color_line.keys(), self.color_line_of_point.keys())
        # for cl in points_cluster_to_line:
        #     self.show_lines_points([cl], points_cluster_to_line[cl], "clusters")

        print "Transfroming...",
        self.adjust_color()
        dir_name = DIR_NAME + self.img.img_name
        cv2.imwrite(dir_name+"/result.bmp", self.gray_after_adjust)
        cv2.imshow("result", self.gray_after_adjust)
        cv2.waitKey()
        print "done"

if __name__ == "__main__":
    files = [
        # "_img/man.bmp",
        # "_img/bird.bmp",
        # "_img/nasu.bmp",
        "_img/teapot.png",
        "_img/teapot0.png",
        "_img/sp5080.png",
        "_img/sp6.png",
        "_img/cylinder.jpg"
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