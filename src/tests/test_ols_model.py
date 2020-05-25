import unittest
import pandas as pd
import numpy as np
import os
import numpy.testing as npt

from ..ols_model import OLSModel

expected_residuals = [
    5.755597,
    -3.153163,
    1.182801,
    -0.297812,
    -1.858975,
    -3.714086,
    1.098721,
    -0.238616,
    -0.198748,
    1.738925,
    2.436941,
    -4.849218,
    3.319369,
    0.136062,
    -2.513857,
    -2.107812,
    -0.953937,
    -0.321683,
    -0.221087,
    -0.539087,
    4.666149,
    -9.816015,
    3.636086,
    4.378233,
    2.496747,
    4.579032,
    -3.720486,
    -1.289573,
    1.980715,
    -1.611223,
]


class TestModel(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(
            f"{os.path.dirname(os.path.abspath(__file__))}/ols_fixture.csv",
            sep=" ",
        )
        self.y = self.df.pop("q")
        self.model = OLSModel(self.df, self.y)

    def test_fit_calculates_predicted_parameters(self):
        self.model.fit()

        npt.assert_array_almost_equal(
            np.array([82.1587, -23.7426, -4.07741, 12.9243, 0.00199456]),
            self.model.a,
            decimal=4,
        )

    def test_fit_calculates_residuals_correctly(self):
        self.model.fit()

        npt.assert_array_almost_equal(
            np.array(expected_residuals), self.model.residuals, decimal=4,
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

    def test_r_squared_significance_high_palue(self):
        model = OLSModel(self.df[["pr"]], self.y)
        model.fit()

        result = model.r_squared_significance()

        self.assertAlmostEqual(0.014102, result.pvalue, places=4)

    def test_r_squared_significance_low_pvalue(self):
        self.model.fit()

        result = self.model.r_squared_significance()

        self.assertAlmostEqual(0, result.pvalue, places=4)

    def test_model_coincidence(self):
        self.model.fit()

        result = self.model.model_coincidence()

        self.assertEqual(2, len(result.coincidence_errors))
        self.assertTrue(dict(variable="pr") in result.coincidence_errors)
        self.assertTrue(dict(variable="i") in result.coincidence_errors)

    def test_jarque_bera_test(self):
        self.model.fit()

        result = self.model.jarque_bera_test()

        self.assertAlmostEqual(0.240498, result.pvalue, places=4)

    def test_runs_test(self):
        self.model.fit()

        result = self.model.runs_test()

        self.assertAlmostEqual(0.390826, result.pvalue, places=4)

    def test_chow_test(self):
        self.model.fit()

        result = self.model.chow_test()

        self.assertAlmostEqual(0.33407, result.pvalue, places=4)

    def test_collinearity_test(self):
        self.model.fit()

        result = self.model.collinearity_test()

        self.assertEqual(3, len(result.collinear_variables))

    def test_breusch_godfrey_test(self):
        self.model.fit()

        result = self.model.breusch_godfrey_test()

        self.assertAlmostEqual(0.07163, result.pvalue, places=4)

    def test_breusch_pagan_test(self):
        self.model.fit()

        result = self.model.breusch_pagan_test()

        self.assertAlmostEqual(0.6593, result.pvalue, places=4)
