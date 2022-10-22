import os

from utils import save_result
from return_code import RC
from recognition import find_items, find_polygon
from packer import predict


def check_image(path_to_items: str, path_to_polygon: str) -> bool:
    #  проверяем, существуют ли переданные файлы
    if not os.path.exists(path_to_items) or not os.path.exists(path_to_polygon):
        print(RC.RC_ENGINE_FILE_ERROR)
        return False
    else:
        # распознаем предметы
        items = find_items(path_to_items)
        # распознаем многоугольник
        polygon = find_polygon(path_to_polygon)
        # делаем вывод об упаковке предметов
        answer = predict(items, polygon)
        # сохраняем результат в csv файл
        save_result(items, polygon, answer)
    return answer
