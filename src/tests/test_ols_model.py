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
            self.model.a,
            np.array([82.1587, -23.7426, -4.07741, 12.9243, 0.00199456]),
            decimal=4,
        )

    def test_add_const(self):
        self.model.add_const()

        npt.assert_equal(
            self.model.col_names, np.array(["const", "pb", "pl", "pr", "i"])
        )
        self.assertEqual(self.model.x.sum(axis=0)[0], len(self.model.x))

    def test_fit_calculates_rsquared(self):
        self.model.fit()

        self.assertAlmostEqual(self.model.rsquare, 0.822118, places=4)

    def test_fit_calculates_adjusted_rsquared(self):
        self.model.fit()

        self.assertAlmostEqual(self.model.rsquare_adjusted, 0.793657, places=4)
