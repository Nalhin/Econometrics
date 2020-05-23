import unittest
import pandas as pd
import numpy as np
import os
import numpy.testing as npt

from ..ols_model import OLSModel


class TestModel(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(
            f"{os.path.dirname(os.path.abspath(__file__))}/ols_fixture.csv",
            sep=" ",
        )
        y = df.pop("q")
        self.model = OLSModel(df, y)

    def test_fit_calculates_predicted_parameters(self):
        self.model.fit()

        npt.assert_array_almost_equal(
            np.array([82.1587, -23.7426, -4.07741, 12.9243, 0.00199456]),
            self.model.a,
            decimal=4,
        )

    def test_fit_calculates_rsquared(self):
        self.model.fit()

        self.assertAlmostEqual(0.822118, self.model.rsquare, places=4)

    def test_fit_calculates_adjusted_rsquared(self):
        self.model.fit()

        self.assertAlmostEqual(self.model.rsquare_adjusted, 0.793657, places=4)

    def test_catalysis_effect(self):
        result = self.model.catalysis_effect()

        self.assertEqual(6, len(result.catalysis_pairs))
        self.assertFalse(result.is_passing)

    def test_statistical_significance_of_parameters(self):
        self.model.fit()

        result = self.model.statistical_significance_of_parameters()

        self.assertAlmostEqual(0.3046, result.variables[2]["pvalue"], places=4)
        self.assertFalse(result.is_passing)
