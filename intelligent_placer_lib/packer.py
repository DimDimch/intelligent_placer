from random import choice
from typing import List
from intelligent_placer_lib.utils import Item, Polygon


def predict(items: List[Item], polygon: Polygon) -> bool:
    return choice([True, False])
