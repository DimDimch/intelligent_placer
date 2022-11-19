import os

from utils import save_result
from return_code import RC
from recognition import recognize
from packer import predict


def check_image(path_to_items: str, path_to_polygon: str, draw: bool = False) -> bool:
    #  checking if the files exist
    if not os.path.exists(path_to_items) or not os.path.exists(path_to_polygon):
        print(RC.RC_ENGINE_FILE_ERROR)
        return False
    else:
        # recognize objects and a polygon
        items, polygon = recognize(path_to_items, path_to_polygon, draw)

        # conclude about the packaging of items
        answer = predict(items, polygon)

        # saving the result to a csv file
        save_result(items, polygon, answer)
    return answer
