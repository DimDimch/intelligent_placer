from typing import List
from utils import Item, Polygon, Figure
from enum import Enum
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import math
import collections
from imageio import imread, imsave
from skimage.color import rgb2gray, label2rgb
from skimage.transform import hough_line, hough_line_peaks, warp, AffineTransform
from skimage.feature import canny, corner_harris, corner_peaks, corner_fast, corner_subpix, match_descriptors, ORB
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.segmentation import watershed
from skimage.morphology import binary_closing, binary_erosion
from skimage.measure import ransac
from scipy import ndimage as ndi
import os

class ItemsCollection(Enum):
    cup_holder = Item(name='Подставка', approx_figure=Figure(np.array([])))
    scotch = Item(name='Скотч', approx_figure=Figure(np.array([])))
    toy = Item(name='Игрушка', approx_figure=Figure(np.array([])))
    calculator = Item(name='Калькулятор', approx_figure=Figure(np.array([])))
    usb = Item(name='Флешка', approx_figure=Figure(np.array([])))
    screwdriver = Item(name='Отвертка', approx_figure=Figure(np.array([])))
    car = Item(name='Машинка', approx_figure=Figure(np.array([])))
    drugs = Item(name='Таблетки', approx_figure=Figure(np.array([])))
    badge = Item(name='Значок', approx_figure=Figure(np.array([])))
    scissors = Item(name='Ножницы', approx_figure=Figure(np.array([])))


def find_group_median(contour: np.array, groups: list) -> np.array:
    def find_median_and_del(group: list):
        global contour
        sum_x, sum_y = 0, 0
        for i in group:
            sum_x += contour[i][0][0]
            sum_y += contour[i][0][1]
        return sum_x / len(group), sum_y / len(group)
    fix_contour = np.array([[[0, 0]]])
    for group in groups:
        m_x, m_y = find_median_and_del(group)
        fix_contour = np.concatenate((fix_contour, [[[int(m_x), int(m_y)]]]))
    return fix_contour[1:]


def preprocess(path_to_image: str, draw: bool = False):
    image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 384))
    orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImageBlur = cv2.blur(grayImage, (2, 2))
    edgedImage = cv2.Canny(grayImageBlur, 18, 50)

    allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    allContours = imutils.grab_contours(allContours)
    allContours = sorted(allContours, key=cv2.contourArea, reverse=True)

    max_perimeter = 0
    paper = None
    for i in range(len(allContours)):
        perimeter = cv2.arcLength(allContours[i], True)
        area = cv2.contourArea(allContours[i])
        contour = cv2.approxPolyDP(allContours[i], 0.02 * perimeter, True)
        if max_perimeter < perimeter and area > 100_000:
            max_perimeter = perimeter
            paper = contour

    if paper is None:
        max_perimeter = 0
        paper_index = 0
        for i in range(len(allContours)):
            perimeter = cv2.arcLength(allContours[i], True)
            if max_perimeter < perimeter:
                max_perimeter = perimeter
                paper_index = i
        paper = cv2.approxPolyDP(allContours[paper_index], 0.02 * max_perimeter, True)

    paper = cv2.minAreaRect(paper)
    paper_box = cv2.boxPoints(paper)
    paper_box = np.int0(paper_box)
    cv2.drawContours(image, [paper_box], 0, (0, 255, 0), 2)

    paper_box = paper_box.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(paper_box, axis=1)
    rect[0] = paper_box[np.argmin(s)]
    rect[2] = paper_box[np.argmax(s)]
    diff = np.diff(paper_box, axis=1)
    rect[1] = paper_box[np.argmin(diff)]
    rect[3] = paper_box[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
    widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))

    if draw:
        fig, ax = plt.subplots(1, 4)
        fig.set_figheight(9)
        fig.set_figwidth(16)
        fig.tight_layout()
        ax[0].set_title(path_to_image)
        ax[0].imshow(orig)
        ax[1].set_title('edged original image')
        ax[1].imshow(edgedImage, cmap='gray')
        ax[2].set_title('find sheet of paper')
        ax[2].imshow(image, cmap='gray')
        ax[3].set_title('cropped sheet')
        ax[3].imshow(scan, cmap='gray')

    return scan


def recognize(path_to_items: str, path_to_polygon: str, draw: bool = False) -> (List[Item], Polygon):
    items_image = preprocess(path_to_items, draw)
    polygon_image = preprocess(path_to_polygon, draw)
    return find_items(items_image), find_polygon(polygon_image)


def find_items(items_image: np.array) -> List[Item]:
    result = list()
    result.append(ItemsCollection.toy.value)
    result.append(ItemsCollection.car.value)
    return result


def find_polygon(polygon_image: np.array) -> Polygon:

    result = Polygon(is_convex=True, num_of_edges=5, data=Figure(np.array([])))
    return result
