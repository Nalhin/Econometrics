import math

import numpy as np
import pandas as pd

from operator import itemgetter

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
from .test_results import (
    CatalysisTestResult,
    SignificanceOfVariablesTestResult,
    PValueTestResult,
    CoincidenceTestResult,
    CollinearityTestResult,
)


def sign(x):
    return math.copysign(1, x)


def t_pvalue(stat, df):
    return stats.t.sf(abs(stat), df=df) * 2


def f_pvalue(stat, df):
    return stats.f.sf(stat, dfn=df[0], dfd=df[1])


def chi_square_pvalue(stat, df):
    return stats.chi2.sf(stat, df)


def z_score_pvalue(z_score):
    return stats.norm.sf(abs(z_score)) * 2


def count_runs_series(series):
    prev_sign = sign(series[0])
    n = 1
    for i in series:
        if sign(i) != prev_sign:
            n += 1
            prev_sign = sign(i)
    return n


class OLSModel:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.x_const = np.insert(self.x, 0, values=1, axis=1)
        self.y = np.array(y)
        self.col_names = (
            x.columns
            if hasattr(x, "columns")
            else np.empty(len(self.x[0]), dtype=np.string_)
        )
        self.col_names_const = np.insert(
            self.col_names, 0, values="const", axis=0
        )
        self.k = len(self.x[0])
        self.n = len(self.x)
        self.a = None
        self.rsquare = 0
        self.rsquare_adjusted = 0
        self.s_squared = 0
        self.ssr = 0
        self.d_squared = None
        self.residuals = None
        self.predicted = None

    def fit(self):
        x_t = np.transpose(self.x_const)
        x_t_x_inverse = np.linalg.inv(x_t @ self.x_const)
        self.a = x_t_x_inverse @ (x_t @ self.y)
        self.predicted = self.x_const @ self.a
        self.residuals = self.y - self.predicted
        self.ssr = np.dot(self.residuals, self.residuals)
        self.rsquare = 1 - (self.ssr / ((self.y - self.y.mean()) ** 2).sum())
        self.rsquare_adjusted = self.rsquare - (
            self.k / (self.n - self.k - 1)
        ) * (1 - self.rsquare)

        self.s_squared = self.ssr / (self.n - self.k - 1)
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
        s_dash = math.sqrt(self.ssr / self.n)
        b_1 = ((self.residuals ** 3).sum() / (self.n * (s_dash ** 3))) ** 2
        b_2 = (self.residuals ** 4).sum() / (self.n * (s_dash ** 4))
        jb = self.n * ((b_1 / 6) + ((b_2 - 3) ** 2) / 24)
        p_value = chi_square_pvalue(jb, df=2)
        return PValueTestResult("JB test", p_value)

    def runs_test(self):
        sorted_residuals = [
            zipped[0]
            for zipped in sorted(
                zip(self.residuals, self.y), key=itemgetter(1)
            )
        ]
        n_plus = (np.array(sorted_residuals) >= 0).sum()
        n_minus = (np.array(sorted_residuals) <= 0).sum()
        n_runs = count_runs_series(sorted_residuals)
        mu = ((2 * n_plus * n_minus) / self.n) + 1
        sigma = math.sqrt(((mu - 1) * (mu - 2)) / (self.n - 1))
        z_score = (n_runs - mu) / sigma
        p_value = z_score_pvalue(z_score)
        return PValueTestResult("Runs test", p_value)

    def chow_test(self):
        x_1, x_2 = np.vsplit(self.x, 2)
        y_1, y_2 = np.split(self.y, 2)
        ols_1 = OLSModel(x_1, y_1)
        ols_2 = OLSModel(x_2, y_2)
        ols_1.fit()
        ols_2.fit()
        rsk_1 = ols_1.ssr
        rsk_2 = ols_2.ssr
        rsk = self.ssr
        r_1 = self.k + 1
        r_2 = self.n - 2 * (self.k + 1)
        f_stat = ((rsk - rsk_1 - rsk_2) / (rsk_1 + rsk_2)) * (r_2 / r_1)
        p_value = f_pvalue(f_stat, df=(r_1, r_2))
        return PValueTestResult("Chow test", p_value)

    def collinearity_test(self):
        collinear = []
        for i in range(self.k):
            x = np.delete(self.x, [i], axis=1)
            model = OLSModel(x, self.x[:, i])
            model.fit()
            if model.rsquare > 0.9:
                collinear.append(self.col_names[i])

        return CollinearityTestResult(collinear)

    def breusch_godfrey_test(self):
        x_e = np.insert(
            self.x,
            self.k,
            values=np.append(np.zeros([1]), self.residuals[:-1]),
            axis=1,
        )
        model = OLSModel(x_e, self.residuals)
        model.fit()
        lm = self.n * model.rsquare
        p_value = chi_square_pvalue(lm, df=1)
        return PValueTestResult("Breuch Godfrey test", p_value)

    def breusch_pagan_test(self):
        sigma = self.ssr / self.n
        model = OLSModel(self.x, self.residuals ** 2 - sigma)
        model.fit()
        lm = self.n * model.rsquare
        chi_p = chi_square_pvalue(lm, df=self.k)
        return PValueTestResult("Breuch Pagan test", chi_p)


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
