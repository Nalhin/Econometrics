import unittest

from src.ols_model import f_pvalue, t_pvalue


class TestModel(unittest.TestCase):
    def test_t_pvalue(self):
        t_stat = 1.3232
        df = 32

        result = t_pvalue(t_stat, df)

        self.assertAlmostEqual(0.194864, result, places=3)

    def test_f_pvalue(self):
        f_stat = 1.2332
        df = (50, 22)

        result = f_pvalue(f_stat, df)

        self.assertAlmostEqual(0.301599, result, places=4)
