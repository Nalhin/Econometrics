from geopy.distance import great_circle

from .config import MODEL_COLUMNS, BERLIN_CENTER
from .plotter import Plotter
from .summarizer import Summarizer


def parse_dataset(df):
    df.set_index("id")
    df.columns = [pascalize(col) for col in df.columns]
    df = df[MODEL_COLUMNS]
    clear_price(df)
    add_distance_from_center(df)
    summarizer = Summarizer(df)
    summarizer.generate_summary_stats()
    plotter = Plotter(df)
    plotter.save_figures()
    drop_tables(df)


def pascalize(snake_str):
    return snake_str.replace("_", " ").title().replace(" ", "")


def drop_tables(df):
    df.drop(["Latitude", "Longitude"], axis=1, inplace=True)


def add_distance_from_center(df):
    df["DistanceFromCenter"] = df.apply(
        lambda x: distance_from_center(x.Latitude, x.Longitude), axis=1
    )


def distance_from_center(lat, lon):
    return great_circle(BERLIN_CENTER, (lat, lon)).km


def clear_price(df):
    df.Price = (
        df["Price"].str.replace("$", "").str.replace(",", "").astype(float)
    )
    df.drop(
        df[(df["Price"] > 500) | (df["Price"] == 0)].index,
        axis=0,
        inplace=True,
    )


def drop_missing(df):
    df.dropna(subset=["Bathrooms", "Bedrooms"], inplace=True)
