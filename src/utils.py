import itertools

from geopy.distance import great_circle

from .config import BERLIN_CENTER


def pascalize(snake_str):
    return snake_str.replace("_", " ").title().replace(" ", "")


def distance_from_center(lat, lon):
    return great_circle(BERLIN_CENTER, (lat, lon)).km


def all_combinations(in_list):
    return [
        c
        for i in range(len(in_list))
        for c in itertools.combinations(in_list, i + 1)
    ]
