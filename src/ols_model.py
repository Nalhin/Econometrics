import math

import pandas as pd
from scipy import stats
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

from .test_results import (
    CatalysisTestResult,
    SignificanceOfVariablesTestResult,
    PValueTestResult,
    CoincidenceTestResult,
)


def sign(x):
    return math.copysign(1, x)


def t_pvalue(stat, df):
    return stats.t.sf(abs(stat), df=df) * 2


def f_pvalue(stat, df):
    return stats.f.sf(stat, dfn=df[0], dfd=df[1])


def chi_square_pvalue(stat, df):
    return stats.chi2.sf(stat, df)


class OLSModel:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.x_const = np.insert(self.x, 0, values=1, axis=1)
        self.y = np.array(y)
        self.col_names = x.columns
        self.col_names_const = np.insert(
            self.col_names, 0, values="const", axis=0
        )
        self.k = len(self.x[0])
        self.n = len(self.x)
        self.a = None
        self.rsquare = 0
        self.rsquare_adjusted = 0
        self.s_squared = 0
        self.d_squared = None
        self.residuals = None

    def fit(self):
        x_t = np.transpose(self.x_const)
        x_t_x_inverse = np.linalg.inv(x_t @ self.x_const)
        self.a = x_t_x_inverse @ (x_t @ self.y)

        y_hat = self.x_const @ self.a
        self.residuals = self.y - y_hat
        e_t_e = (self.residuals ** 2).sum()
        self.rsquare = 1 - (e_t_e / ((self.y - self.y.mean()) ** 2).sum())
        self.rsquare_adjusted = self.rsquare - (
            self.k / (self.n - self.k - 1)
        ) * (1 - self.rsquare)
        self.s_squared = e_t_e / (self.n - self.k - 1)
        self.d_squared = self.s_squared * x_t_x_inverse

    def catalysis_effect(self):
        r = np.corrcoef(self.x.transpose())
        r_0 = [np.corrcoef(self.x[:, i], self.y)[1][0] for i in range(self.k)]
        pairs = []
        for i in range(self.k):
            for j in range(self.k):
                if abs(r_0[i]) < abs(r_0[j]):
                    r_ij = r[i, j] * sign(r_0[i]) * sign(r_0[j])
                    if r_ij > (abs(r_0[i]) / abs(r_0[j])) or r_ij < 0:
                        pairs.append((self.col_names[i], self.col_names[j]))

        return CatalysisTestResult(pairs)

    def statistical_significance_of_parameters(self):
        results = []
        for i in range(self.k + 1):
            s_a = math.sqrt(self.d_squared[i, i])
            a_j = self.a[i]

            t_calculated = a_j / s_a
            pvalue = t_pvalue(t_calculated, df=self.n - self.k - 1)
            results.append(
                dict(
                    pvalue=pvalue,
                    variable=self.col_names_const[i],
                    tstat=t_calculated,
                )
            )

        return SignificanceOfVariablesTestResult(results)

    def model_coincidence(self):
        r_0 = [np.corrcoef(self.x[:, i], self.y)[0][1] for i in range(self.k)]
        a = self.a[1:]
        results = []
        for r in range(self.k):
            if sign(r_0[r]) != sign(a[r]):
                results.append(dict(variable=self.col_names[r]))

        return CoincidenceTestResult(results)

    def r_squared_significance(self):
        f_calculated = (
            (self.rsquare / self.k)
            * (self.n - self.k - 1)
            / (1 - self.rsquare)
        )
        p_value = f_pvalue(f_calculated, df=(self.k, self.n - self.k - 1))
        return PValueTestResult("The significance of r squared", p_value)

    def jarque_bera_test(self):
        s_dash = math.sqrt((self.residuals ** 2).sum() / self.n)
        b_1 = ((self.residuals ** 3).sum() / (self.n * (s_dash ** 3))) ** 2
        b_2 = (self.residuals ** 4).sum() / (self.n * (s_dash ** 4))
        jb = self.n * ((b_1 / 6) + ((b_2 - 3) ** 2) / 24)
        p_value = chi_square_pvalue(jb, df=2)
        return PValueTestResult("JB test", p_value)


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
