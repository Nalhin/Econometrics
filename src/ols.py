import math

import numpy as np

from operator import itemgetter
from scipy import stats

from .stat_tests_results import (
    CatalysisTestResult,
    SignificanceOfParametersTestResult,
    PValueTestResult,
    CoincidenceTestResult,
    CollinearityTestResult,
    Estimator,
    TestType,
)


def sign(x):
    return math.copysign(1, x)


def t_p_value(stat, df):
    return stats.t.sf(abs(stat), df=df) * 2


def f_p_value(stat, df):
    return stats.f.sf(stat, dfn=df[0], dfd=df[1])


def chi_square_p_value(stat, df):
    return stats.chi2.sf(stat, df)


def z_score_p_value(z_score):
    return stats.norm.sf(abs(z_score)) * 2


def count_runs_series(series):
    prev_sign = sign(series[0])
    n = 1
    for i in series:
        if sign(i) != prev_sign:
            n += 1
            prev_sign = sign(i)
    return n


class OLS:
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
        self.predicted = self.predict(self.x_const)
        self.residuals = self.y - self.predicted
        self.ssr = np.dot(self.residuals, self.residuals)
        self.rsquare = 1 - (self.ssr / ((self.y - self.y.mean()) ** 2).sum())
        self.rsquare_adjusted = self.rsquare - (
            self.k / (self.n - self.k - 1)
        ) * (1 - self.rsquare)

        self.s_squared = self.ssr / (self.n - self.k - 1)
        self.d_squared = self.s_squared * x_t_x_inverse

    def predict(self, x):
        return x @ self.a

    def get_validation_results(self):
        tests = {}
        for t in [
            self.catalysis_effect(),
            self.statistical_significance_of_parameters(),
            self.model_coincidence(),
            self.r_squared_significance(),
            self.jarque_bera_test(),
            self.runs_test(),
            self.chow_test(),
            self.collinearity_test(),
            self.breusch_godfrey_test(),
            self.breusch_pagan_test(),
            self.ramsey_reset(),
        ]:
            tests[t.type] = t

        return tests

    @property
    def is_statistically_correct(self):
        return all(
            test.is_passing for test in self.get_validation_results().values()
        )

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
            s_a = math.sqrt(abs(self.d_squared[i, i]))
            a_j = self.a[i]
            t_calculated = a_j / s_a
            p_value = t_p_value(t_calculated, df=self.n - self.k - 1)
            results.append(
                dict(
                    p_value=p_value,
                    variable=self.col_names_const[i],
                    t_stat=t_calculated,
                )
            )

        return SignificanceOfParametersTestResult(results)

    def model_coincidence(self):
        r_0 = [np.corrcoef(self.x[:, i], self.y)[0][1] for i in range(self.k)]
        a = self.a[1:]
        results = []
        for r in range(self.k):
            if sign(r_0[r]) != sign(a[r]):
                results.append(
                    dict(variable=self.col_names[r], a=a[r], r=r_0[r])
                )

        return CoincidenceTestResult(results, summary=zip(a, r_0))

    def r_squared_significance(self):
        f_calculated = (
            (self.rsquare / self.k)
            * (self.n - self.k - 1)
            / (1 - self.rsquare)
        )
        p_value = f_p_value(f_calculated, df=(self.k, self.n - self.k - 1))
        return PValueTestResult(
            TestType.R_SQUARE_SIGNIFICANCE, p_value, f_calculated, smaller=True
        )

    def jarque_bera_test(self):
        s_dash = math.sqrt(self.ssr / self.n)
        b_1 = ((self.residuals ** 3).sum() / (self.n * (s_dash ** 3))) ** 2
        b_2 = (self.residuals ** 4).sum() / (self.n * (s_dash ** 4))
        jb = self.n * ((b_1 / 6) + ((b_2 - 3) ** 2) / 24)
        p_value = chi_square_p_value(jb, df=2)
        return PValueTestResult(
            TestType.JARQUE_BERA, p_value, jb, optional=True
        )

    def runs_test(self):
        sorted_residuals = [
            zipped[0]
            for zipped in sorted(
                zip(self.residuals, self.y), key=itemgetter(1)
            )
        ]
        n_plus = (np.array(sorted_residuals) > 0).sum()
        n_minus = (np.array(sorted_residuals) < 0).sum()
        n_runs = count_runs_series(sorted_residuals)
        mu = ((2 * n_plus * n_minus) / self.n) + 1
        sigma = math.sqrt(((mu - 1) * (mu - 2)) / (self.n - 1))
        z_score = (n_runs - mu) / sigma if sigma else math.inf
        p_value = z_score_p_value(z_score)
        return PValueTestResult(TestType.RUNS, p_value, z_score)

    def chow_test(self):
        x_1, x_2 = np.vsplit(self.x, [self.n // 2])
        y_1, y_2 = np.split(self.y, [self.n // 2])
        ols_1 = OLS(x_1, y_1)
        ols_2 = OLS(x_2, y_2)
        ols_1.fit()
        ols_2.fit()
        rsk_1 = ols_1.ssr
        rsk_2 = ols_2.ssr
        rsk = self.ssr
        r_1 = self.k + 1
        r_2 = self.n - 2 * (self.k + 1)
        f_stat = ((rsk - rsk_1 - rsk_2) / (rsk_1 + rsk_2)) * (r_2 / r_1)
        p_value = f_p_value(f_stat, df=(r_1, r_2))
        return PValueTestResult(TestType.CHOW, p_value, f_stat)

    def collinearity_test(self):
        collinear = []
        for i in range(self.k):
            x = np.delete(self.x, [i], axis=1)
            model = OLS(x, self.x[:, i])
            model.fit()
            if model.rsquare > 0.9:
                collinear.append(self.col_names[i])

        return CollinearityTestResult(collinear)

    def breusch_godfrey_test(self):
        x_e = np.c_[self.x, np.append(np.zeros([1]), self.residuals[:-1])]
        model = OLS(x_e, self.residuals)
        model.fit()
        lm = self.n * model.rsquare
        p_value = chi_square_p_value(lm, df=1)
        return PValueTestResult(TestType.BREUCH_GODFREY, p_value, lm)

    def breusch_pagan_test(self):
        sigma = self.ssr / self.n
        model = OLS(self.x, self.residuals ** 2 - sigma)
        model.fit()
        lm = self.n * model.rsquare
        chi_p = chi_square_p_value(lm, df=self.k)
        return PValueTestResult(TestType.BREUCH_PAGAN, chi_p, lm)

    def ramsey_reset(self):
        extended_x = np.c_[self.x, self.predicted ** 2, self.predicted ** 3]
        model = OLS(extended_x, self.y)
        model.fit()
        rs_1 = self.rsquare
        rs_2 = model.rsquare
        df_1 = 2
        df_2 = self.n - self.k - 3
        reset = (rs_2 - rs_1) / (1 - rs_2) * (df_2 / df_1)
        p_value = f_p_value(reset, df=(df_1, df_2))
        return PValueTestResult(TestType.RAMSEY_RESET, p_value, reset)

    def get_prediction_errors(self, x, y):
        x = np.insert(x, 0, values=1, axis=1)
        y = np.array(y)
        predicted = self.predict(x)
        diff = y - predicted
        me = diff.mean()
        mae = abs(diff).mean()
        rmse = math.sqrt((diff ** 2).mean())
        mape = abs(diff / y).mean() * 100
        return Estimator(me, mae, rmse, mape)

    def transform_heteroskedacity(self):
        self.x = self.x / np.sqrt(self.residuals ** 2).reshape(-1, 1)
        self.y = self.y / np.sqrt(self.residuals ** 2)
        self.x_const = np.insert(self.x, 0, values=1, axis=1)
