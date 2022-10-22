from dataclasses import dataclass
from typing import List

import numpy as np


"""
Данный класс является основой для работы со всеми многоугольниками в программе.
По своей сути он представляет обертку над классом polytope из библиотеки polytope
"""


class Figure:
    def __init__(self, fig: np.array):
        pass


"""
Данный класс хранит в себе информацию о распознанном предмете (все возможные предметы заранее
известны и находятся в каталоге items)
"""


@dataclass(eq=True)
class Item:
    name: str  # уникальное наименование предмета
    approx_figure: Figure


"""
Данный класс хранит в себе информацию о распознанном многоугольнике
"""


@dataclass(eq=True)
class Polygon:
    is_convex: bool  # является ли выпуклым
    num_of_edges: int  # примерное число ребер
    data: Figure


"""
Данный метод делает pretty print and save всех результатов эксперимента
"""


def save_result(items: List[Item], polygon: Polygon, answer: bool):
    with open('results.csv', 'a') as f:
        f.write('')
    pass