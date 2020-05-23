import pandas as pd

from .parse_dataset import parse_dataset

df_initial = pd.read_csv("./data/listings_summary.csv")

parse_dataset(df_initial)
