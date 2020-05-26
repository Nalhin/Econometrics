import pandas as pd

from .ols_estimator import OLSEstimator

for i in range(10):
    ols = OLSEstimator(pd.read_csv("./data/listings_summary.csv"))
    ols.clean_data()
    ols.calculate_models()
    ols.output_latex()
