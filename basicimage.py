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
        # if img_name is a string, create new instance; else if it's an object, copy it
        if isinstance(img_name, str):
            print("Reading _img file '" + img_name + "'...\n")
            self.img_name = img_name.split(".")[0]
            self.img = cv2.imread("./" + img_name)
        elif isinstance(img_name, Input):
            self.img = img_name.copy()
        else:
            print("Illegle parameter!")
            return
        self.rows, self.cols, dim = self.img.shape
        cols = self.cols
        rows = self.rows

        # RGB scale
        self.bgr = self.img
        # gray scale
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # hsl scale
        self.hls = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)

        # some correspondances
        # self.pixels_points = {}  # pixel:point
        # self.points_pixels = {}  # point:pixels
        self.points_norms = {}   # point:norm

        # some type of pixles
        self.pixels = {}
        self.points = {}
        self.fg_pixels = {}
        self.bg_pixels = {}
        self.fg_points = {}
        self.bg_points = {}

        # image to show background and foreground regions
        self.fg_map = np.ndarray((rows, cols), dtype=np.uint8)
        self.fg_map_show = self.fg_map.copy()

        # generate fg_map
        for pixel in [(r, c) for r in range(self.rows) for c in range(self.cols)]:
            blue, green, red = point = self.get_bgr_value(pixel)
            if point not in self.points_norms:
                self.points_norms[point] = self.get_bgr_norm(point)

            self.pixels[pixel] = 1
            self.points[point] = 1
            if blue >= 250 and green >= 250 and red >= 250:  # or blue < 5 and green < 5 and red < 5:
                self.fg_map.itemset(pixel, 0)
                self.fg_map_show.itemset(pixel, 0)
                self.bg_pixels[pixel] = "1"
                self.bg_points[point] = "1"
            else:
                self.fg_map.itemset(pixel, 1)
                self.fg_map_show.itemset(pixel, 255)
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

    def resize_fg_gray(self, img):
        gray_xs, gray_ys = zip(*self.fg_pixels.keys())
        x_max = max(gray_xs)
        y_max = max(gray_ys)
        x_min = min(gray_xs)
        y_min = min(gray_ys)
        # new_img = np.zeros((x_max-x_min+1, y_max-y_min+1), dtype=np.uint8)
        new_img = np.array([255]*(x_max-x_min+1)*(y_max-y_min+1), dtype=np.uint8).reshape((x_max-x_min+1, y_max-y_min+1))
        for p in self.fg_pixels:
            x, y = p
            x = x - x_min
            y = y - y_min
            new_img.itemset((x, y), img.item(p))
        return new_img

    def show_fg(self):
        cv2.imshow(self.img_name + "_fg", self.fg_map_show)

    @staticmethod
    def draw_hemi(ax, slice_number, slice_width, slice_start_norm):
        for i in range(slice_number):
            r = slice_width * i + slice_start_norm
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
            points_rgb = zip(r, g, b)
            if cmap is None:
                color_map = map(lambda x: map(lambda y: y / float(255) if 255 >= y >= 0 else 1, x), points_rgb)
                # bs, gs, rs = color_map
                # color_map = [rs, gs, bs]
            else:
                color_map = cmap
            ax.scatter(b, g, r, s=30, c=color_map, marker="o", edgecolors='None')
            # @staticmethod
            # def fig_draw_line(ax,lines):