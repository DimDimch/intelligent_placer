from dataclasses import dataclass
from typing import List
import numpy as np


"""
This class is the basis for working with all polygons in the program.
At its core, it represents a wrapper over the polytope class from the polytope library
"""


class Figure:
    def __init__(self, fig: np.array):
        pass


"""
This class stores information about the recognized item (all possible items
are known in advance and are located in the items catalog)
"""


@dataclass(eq=True)
class Item:
    name: str  # unique item name
    approx_figure: Figure


"""
This class stores information about the recognized polygon
"""


@dataclass(eq=True)
class Polygon:
    is_convex: bool
    num_of_edges: int  # approximate number of edges
    data: Figure


"""
This method makes a pretty print and save of all the results of the test
"""


def save_result(items: List[Item], polygon: Polygon, answer: bool):
    with open('results.csv', 'a') as f:
        f.write('')
    pass
