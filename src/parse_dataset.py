import itertools

from geopy.distance import great_circle

from .config import BERLIN_CENTER


def pascalize(snake_str):
    return snake_str.replace("_", " ").title().replace(" ", "")


def add_distance_from_center(df):
    df["DistanceFromCenter"] = df.apply(
        lambda x: distance_from_center(x.Latitude, x.Longitude), axis=1
    )


def distance_from_center(lat, lon):
    return great_circle(BERLIN_CENTER, (lat, lon)).km


def clean_price(df):
    df.Price = (
        df["Price"].str.replace("$", "").str.replace(",", "").astype(float)
    )
    df.drop(
        df[(df["Price"] > 500) | (df["Price"] == 0)].index,
        axis=0,
        inplace=True,
    )


def all_combinations(in_list):
    return [
        c
        for i in range(len(in_list))
        for c in itertools.combinations(in_list, i + 1)
    ]
