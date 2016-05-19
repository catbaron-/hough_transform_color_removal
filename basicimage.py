import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Input:
    """
    Class for input image
    """

    def __init__(self, img_name):
        # read image
        if isinstance(img_name, str):
            print("Reading img file '" + img_name + "'...\n")
            self.img_name = img_name.split(".")[0]
            self.img = cv2.imread("./" + img_name)
        else:
            self.img = img_name.copy()
        self.rows, self.cols, dim = self.img.shape
        cols = self.cols
        rows = self.rows


        # RGB scale
        self.bgr = self.img
        # gray scale
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # hsl scale
        self.hls = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)

        # location list for all pixels
        self.pixels_points = {}
        self.points_pixels = {}
        self.points_norms = {}
        for pixel in [(r, c) for r in range(self.rows) for c in range(self.cols)]:
            point = self.get_bgr_value(pixel)
            self.pixels_points[pixel] = point
            if point not in self.points_pixels:
                self.points_pixels[point] = [pixel]
                self.points_norms[point] = self.get_bgr_norm(point)
            else:
                self.points_pixels[point].append(pixel)
        self.fg_pixels = {}
        self.bg_pixels = {}
        self.fg_points = {}
        self.bg_points = {}

        # image to show background and foreground regions
        self.fg_map = np.ndarray((rows, cols), dtype=np.uint8)
        self.fg_map_show = self.fg_map.copy()

        # generate fg_map
        for pixel in self.pixels_points:
            blue, green, red = self.get_bgr_value(pixel)
            if blue >= 250 and green >= 250 and red >= 250:
                self.fg_map.itemset(pixel, 0)
                self.fg_map_show.itemset(pixel, 0)
            else:
                self.fg_map.itemset(pixel, 1)
                self.fg_map_show.itemset(pixel, 255)

        print "Doing Segmentation..."
        # pixels of #FFF is background, labeled as 0
        for pixel in self.pixels_points:
            b, g, r = point = self.get_bgr_value(pixel)
            if b >= 253 and g >= 253 and r >= 253:
                self.bg_pixels[pixel] = "1"
                self.bg_points[point] = "1"
            else:
                self.fg_pixels[pixel] = "1"
                self.fg_points[point] = "1"

    def reduce_color(self, m=3):
        img_reduce = self.img.copy()
        for pixel in self.fg_pixels:
            x, y = pixel
            b, g, r = self.get_bgr_value(pixel)
            bgr = map(lambda x: int(x) / m * m, [b, g, r])
            img_reduce[x][y] = np.array(bgr)
        return img_reduce

    @staticmethod
    def linear_lumiance(point):
        b, g, r = point
        return 0.21 * r + 0.73 * g + 0.06 * b

    @staticmethod
    def get_bgr_norm(point):
        p_arr = np.array(point)
        return np.sqrt(float(p_arr.dot(p_arr)))

    def get_bgr_value(self, point_xy, bgr_map=None):
        if bgr_map is None:
            bgr_map = self.img
        bgr = bgr_map[point_xy[0]][point_xy[1]]
        return float(bgr[0]), float(bgr[1]), float(bgr[2])

    def is_background(self, p):
        if 1 > self.fg_map.item(p):
            return True
        else:
            return False

    def show_fg(self):
        cv2.imshow(self.img_name + "_fg", self.fg_map_show)

    @staticmethod
    def draw_hemi(ax, slice_number, slice_width):
        for i in range(slice_number):
            r = slice_width * (i + 1)
            x = np.arange(0, r, r / 10)
            y = np.arange(0, r, r / 10)
            # z = np.sqrt(np.add(np.multiply(x, x), np.multiply(y, y)))
            x, y = np.meshgrid(x, y)
            redio = r ** 2 - x ** 2 - y ** 2

            z = np.sqrt(redio)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.06)

    @staticmethod
    def fig_set_label(ax, x="Blue", y="Green", z="Red", limit=300):
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_xlim([0, limit])
        ax.set_ylim([0, limit])
        ax.set_zlim([0, limit])

    @staticmethod
    def fig_draw_lines(ax, lines):
        for line in lines:
            z, y, x = list(zip(*line))
            ax.plot(z, y, x)

    @staticmethod
    def fig_draw_points(ax, points, cmap=None):
        if points:
            b, g, r = zip(*points)
            if cmap == None:
                color_map = map(lambda x: map(lambda y: y / float(255) if 255 > y > 0 else 1, x), points)
            else:
                color_map = cmap
            ax.scatter(b, g, r, s=30, c=color_map, marker="o", edgecolors='None')
            # @staticmethod
            # def fig_draw_line(ax,lines):
