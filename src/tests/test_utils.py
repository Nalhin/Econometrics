import unittest

from ..ols import (
    f_p_value,
    t_p_value,
    chi_square_p_value,
    z_score_p_value,
)


class TestUtils(unittest.TestCase):
    def test_t_pvalue(self):
        t_stat = 1.3232
        df = 32

        result = t_p_value(t_stat, df)

        self.assertAlmostEqual(0.194864, result, places=3)

    def test_f_pvalue(self):
        f_stat = 1.2332
        df = (50, 22)

        result = f_p_value(f_stat, df)

        self.assertAlmostEqual(0.301599, result, places=4)

    def test_chi_square_pvalue(self):
        chi_square_stat = 2.222
        df = 5

        result = chi_square_p_value(chi_square_stat, df)

        self.assertAlmostEqual(0.817652, result, places=4)

    def test_z_score_pvalue(self):
        z_score = 1.21

        result = z_score_p_value(z_score)

        self.assertAlmostEqual(0.226279, result, places=4)
