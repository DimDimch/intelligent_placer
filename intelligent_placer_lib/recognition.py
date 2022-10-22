from typing import List
from utils import Item, Polygon, Figure
from enum import Enum
import numpy as np


class ItemsCollection(Enum):
    cup_holder = Item(name='Подставка', approx_figure=Figure(np.array([])))
    scotch = Item(name='Скотч', approx_figure=Figure(np.array([])))
    toy = Item(name='Игрушка', approx_figure=Figure(np.array([])))
    calculator = Item(name='Калькулятор', approx_figure=Figure(np.array([])))
    usb = Item(name='Флешка', approx_figure=Figure(np.array([])))
    screwdriver = Item(name='Отвертка', approx_figure=Figure(np.array([])))
    car = Item(name='Машинка', approx_figure=Figure(np.array([])))
    drugs = Item(name='Подставка', approx_figure=Figure(np.array([])))
    badge = Item(name='Значок', approx_figure=Figure(np.array([])))
    scissors = Item(name='Ножницы', approx_figure=Figure(np.array([])))


def find_items(path: str) -> List[Item]:
    result = list()
    result.append(ItemsCollection.toy.value)
    result.append(ItemsCollection.car.value)
    return result


def find_polygon(path: str) -> Polygon:
    result = Polygon(is_convex=True, num_of_edges=5, data=Figure(np.array([])))
    return result
