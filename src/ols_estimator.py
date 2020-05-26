import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.model_selection import train_test_split

from .config import MODEL_COLUMNS, FILTERED_COLUMNS
from .model_summary import ModelSummary
from .ols import OLS
from .plotter import Plotter
from .summarizer import Summarizer
from .utils import pascalize, distance_from_center, all_combinations


class OLSEstimator:
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
        self.df = self.df[MODEL_COLUMNS]
        self.df.set_index("Id", inplace=True)
        self.df.RoomType.astype("category")
        self.df.Price = (
            self.df["Price"]
            .str.replace("$", "")
            .str.replace(",", "")
            .astype(float)
        )
        self.add_calculated()
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
        self.df["DistanceFromCenter"] = self.df.apply(
            lambda x: distance_from_center(x.Latitude, x.Longitude), axis=1
        )

    def remove_outliers(self):
        df = self.df.copy(deep=True)
        for col in FILTERED_COLUMNS:
            df = df[(np.abs(stats.zscore(df[col])) < 3)]
        self.df = df

    def split(self):
        df = pd.get_dummies(self.df, drop_first=True, prefix_sep="")
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
        combinations = list(combinations)

        for com in combinations:
            model = OLS(self.x_train[list(com)], self.y_train)
            model.fit()
            try:
                summary = ModelSummary(
                    model.get_validation_results(),
                    list(com),
                    model.a,
                    model.get_prediction_errors(
                        self.x_test[list(com)].to_numpy(),
                        self.y_test.to_numpy(),
                    ),
                    model.rsquare,
                )
                if not model.breusch_pagan_test().is_passing:
                    model.transform_heteroskedacity()
                    model.fit()
                    summary = ModelSummary(
                        model.get_validation_results(),
                        list(com),
                        model.a,
                        model.get_prediction_errors(
                            self.x_test[list(com)].to_numpy(),
                            self.y_test.to_numpy(),
                        ),
                        model.rsquare,
                    )
                    if model.breusch_pagan_test().is_passing:
                        self.models.append(summary)
                else:
                    self.models.append(summary)
            except Exception as e:
                print(e)
        for model in self.models:
            model.to_latex()
