from pygemstones.util import log as l

from modules.cl_recommendation import Recommendation


def show_recommendations(title: str, list: [Recommendation]):
    l.colored(title, l.MAGENTA)

    for item in list:
        if item.distance:
            l.colored(f"{item.title} - ({item.distance:.2f})", l.GREEN)
        else:
            l.colored(f"{item.title}", l.GREEN)
