
from typing import List
import numpy as np
from intelligent_placer_lib.recognition import Item
from shapely.geometry import Polygon
from shapely.affinity import translate
from shapely.plotting import plot_polygon
from rtree import index


def predict(items: List[Item], polygon: Polygon, draw: bool = False, ax=None) -> bool:
    packer = Packer(polygon)
    res = packer.pack(items)
    if draw:
        packer.draw(ax)
    return res


class Packer:

    def __init__(self, polygon: Polygon):
        self.polygon = polygon

        self.packedRTree = index.Index()

        self.init_split_size = 100
        self.marching_max_iter = 100

    def generate_init_places(self, geom):
        bounds = self.polygon.bounds
        top_left = np.array([bounds[0], bounds[3]])

        bound_width = bounds[2] - bounds[0]
        step = bound_width / self.init_split_size

        geom_centroid = np.array(geom.centroid.coords[0])
        bias = top_left - geom_centroid
        geom = translate(geom, xoff=bias[0], yoff=bias[1])

        for i in range(self.init_split_size + 1):
            geom = translate(geom, xoff=step, yoff=0.0)
            yield geom

    def pack(self, geom_list: List[Item]):
        geom_list = [item.approx_figure for item in geom_list]
        geom_list.sort(key=lambda poly: poly.area, reverse=True)

        for geom in geom_list:
            pack_success = False

            for init_geom in self.generate_init_places(geom):
                placed = self.bottom_left_search(init_geom)

                if not placed is None:
                    self.add_geom(placed)
                    pack_success = True
                    break

            if not pack_success:
                return False

        return True

    def marching_steps(self, geom):
        bounds = self.polygon.bounds
        bottom_left = np.array([bounds[0], bounds[1]])

        geom_centroid = np.array(geom.centroid.coords[0])
        steps = (bottom_left - geom_centroid) / (self.marching_max_iter - 1)
        return steps

    def bottom_left_search(self, geom):
        steps = self.marching_steps(geom)
        shife_left = lambda geom: translate(geom, xoff=steps[0])
        shife_bottom = lambda geom: translate(geom, yoff=steps[1])

        place_found = False
        while True:
            moved = False

            geom, step = self.marching(geom, shife_bottom)
            if step >= 0:
                moved = True

            geom, step = self.marching(geom, shife_left)
            if step >= 0:
                moved = True

            if not moved:
                break

            place_found = True

        if place_found:
            return geom
        return None

    def add_geom(self, geom):
        id = self.packedRTree.get_size()
        self.packedRTree.insert(id, geom.bounds, obj=geom)

    def marching(self, geom, translate):
        result_step = -1
        result = geom

        for i in range(self.marching_max_iter):
            geom = translate(geom)
            if not self.checkOverlap(geom):
                result = geom
                result_step = i

        return result, result_step

    def checkOverlap(self, geom):
        if not geom.within(self.polygon):
            return True

        for n in self.packedRTree.intersection(geom.bounds, objects=True):
            if n.object.intersects(geom):
                return True

        return False

    def draw(self, ax):
        bounds = self.polygon.bounds
        plot_polygon(self.polygon, ax=ax[2], add_points=False)

        x_margin = 0.1 * (bounds[2] - bounds[0])
        y_margin = 0.1 * (bounds[3] - bounds[1])

        left = bounds[0] - x_margin
        right = bounds[2] + x_margin
        bottom = bounds[1] - y_margin
        top = bounds[3] + y_margin

        for n in self.packedRTree.intersection(bounds, objects=True):
            plot_polygon(n.object, ax=ax[2], add_points=False)

        ax[2].set_xlim([left, right])
        ax[2].set_ylim([bottom, top])
