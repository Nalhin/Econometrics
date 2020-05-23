import pandas as pd
from sklearn.model_selection import train_test_split
from .config import MODEL_COLUMNS
from .parse_dataset import (
    add_distance_from_center,
    clean_price,
    pascalize,
    all_combinations,
)
from .plotter import Plotter
from .summarizer import Summarizer
import numpy as np


class OLSModel:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.col_names = x.columns
        self.a = []
        self.k = len(self.x[0])
        self.n = len(self.x)
        self.rsquare = 0
        self.rsquare_adjusted = 0

    def catalyst_effect(self):
        pass

    def add_const(self):
        self.x = np.insert(self.x, 0, values=1, axis=1)
        self.col_names = np.insert(self.col_names, 0, "const", axis=0)

    def fit(self):
        self.add_const()
        x_t = np.transpose(self.x)
        self.a = np.linalg.inv(x_t @ self.x) @ (x_t @ self.y)
        y_hat = self.x @ self.a
        self.rsquare = 1 - (
            ((self.y - y_hat) ** 2).sum()
            / ((self.y - self.y.mean()) ** 2).sum()
        )
        self.rsquare_adjusted = self.rsquare - (
            self.k / (self.n - self.k - 1)
        ) * (1 - self.rsquare)



class OLS:
    def __init__(self, df):
        self.df = df
        self.models = []
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.black_list = ["Longitude", "Latitude", "Id"]

    def clean_data(self):
        self.df.columns = [pascalize(col) for col in self.df.columns]
        self.df.set_index("Id")
        self.df = self.df[MODEL_COLUMNS]
        clean_price(self.df)
        add_distance_from_center(self.df)
        self.remove_outliers()

    def calculate_models(self):
        self.split()
        self.fit_models()

    def output_latex(self):
        plotter = Plotter(self.df)
        plotter.save_figures()
        summarizer = Summarizer(self.df)
        summarizer.generate_summary_stats()

    def add_calculated(self):
        add_distance_from_center(self.df)

    def remove_outliers(self):
        pass

    def split(self):
        df = pd.get_dummies(self.df, drop_first=True)
        df = df.dropna()
        y = df.pop("Price")
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(df, y, test_size=0.1)

    def fit_models(self):
        combinations = all_combinations(
            list(
                filter(
                    lambda x: x not in self.black_list, self.x_train.columns
                )
            )
        )
        combinations = list(combinations)[15:17]
        for com in combinations:
            model = OLSModel(self.x_train[list(com)], self.y_train)
            model.fit()
            self.models.append(model)
