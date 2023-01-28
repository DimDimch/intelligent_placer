import json
from dataclasses import dataclass
from typing import List
from IPython.core.display import display, Image
from shapely.geometry import shape
from shapely.geometry import Polygon

import numpy as np
import cv2
import imutils
from ultralytics import YOLO


def get_figure(fig_name: str):
    with open('figures.json', 'r') as f:
        data = json.loads(f.read())
    if fig_name != 'scotch':
        return shape(data[fig_name])
    else:
        return shape(data[fig_name]['outside']).difference(shape(data[fig_name]['inside']))


model = YOLO('intelligent_placer_lib/models/best.pt')


@dataclass(eq=True)
class Item:
    name: str  # unique item name
    id: int
    approx_figure: Polygon
    rotate: bool
    meta: dict = None


ItemsCollection = {
    0: Item(name='calculator', id=0, approx_figure=get_figure('calculator'), rotate=True),
    1: Item(name='car', id=1, approx_figure=get_figure('car'), rotate=True),
    2: Item(name='cup_holder', id=2, approx_figure=get_figure('cup_holder'), rotate=False),
    3: Item(name='icon', id=3, approx_figure=get_figure('icon'), rotate=False),
    4: Item(name='scissors', id=4, approx_figure=get_figure('scissors'), rotate=True),
    5: Item(name='scotch', id=5, approx_figure=get_figure('scotch'), rotate=False, meta={'inside': ('icon', 'car', 'toy')}),
    6: Item(name='screwdriver', id=6, approx_figure=get_figure('screwdriver'), rotate=True),
    7: Item(name='tablets', id=7, approx_figure=get_figure('tablets'), rotate=True),
    8: Item(name='toy', id=8, approx_figure=get_figure('toy'), rotate=False),
    9: Item(name='usb', id=9, approx_figure=get_figure('usb'), rotate=True),
}


def find_polygon(path_to_image: str):
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
    scan = cv2.resize(scan, (500, 353))

    grayScan = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    grayScanBlur = cv2.blur(grayScan, (6, 6))
    edgedScan = cv2.Canny(grayScanBlur, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edgedScan, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scan_copy = scan.copy()
    cv2.drawContours(scan_copy, contours, -1, (0, 255, 0), 2)

    polygon = None
    if len(contours) > 0:
        contour = np.squeeze(contours[0])
        polygon = Polygon(contour)
        # polygon = scale(Polygon(contour), xfact=-1, origin=(1, 0))
        # polygon = rotate(polygon, angle=180, origin='center')

    return polygon


def recognize(path_to_items: str, path_to_polygon: str, draw: bool = False, ax=None) -> (List[Item], Polygon):
    # preprocess for items image
    items_image = cv2.imread(path_to_items, cv2.IMREAD_COLOR)
    items_image = cv2.resize(items_image, (640, 640))

    if draw:
        ax[0].imshow(cv2.resize(cv2.imread(path_to_items, cv2.IMREAD_COLOR), (500, 353)))
        ax[1].imshow(cv2.resize(cv2.imread(path_to_polygon, cv2.IMREAD_COLOR), (500, 353)))

    return find_items(items_image), find_polygon(path_to_polygon)


def find_items(items_image: np.array) -> List[Item]:
    results = model.predict(source=items_image, save=False)
    items_labels = list(set(results[0].boxes.cls.tolist()))
    result = []
    for item in items_labels:
        result.append(ItemsCollection[item])
    return result
