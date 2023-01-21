import os

from intelligent_placer_lib.return_code import RC
from intelligent_placer_lib.recognition import recognize
from intelligent_placer_lib.packer import predict
import matplotlib.pyplot as plt


def check_image(path_to_items: str, path_to_polygon: str, draw: bool = False) -> bool:
    #  checking if the files exist
    if not os.path.exists(path_to_items) or not os.path.exists(path_to_polygon):
        print(RC.RC_ENGINE_FILE_ERROR)
        return False
    else:
        ax = None
        if draw:
            fig, ax = plt.subplots(1, 3)
            fig.set_figheight(4)
            fig.set_figwidth(16)
            fig.tight_layout()


        # recognize objects and a polygon
        items, polygon = recognize(path_to_items, path_to_polygon, draw, ax)

        # conclude about the packaging of items
        answer = predict(items, polygon, draw, ax)

        if draw:
            plt.show()
        # saving the result to a csv file
        # save_result(items, polygon, answer)
    return answer
