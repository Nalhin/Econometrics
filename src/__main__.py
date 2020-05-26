import pandas as pd

from .ols_estimator import OLSEstimator

ols = OLSEstimator(pd.read_csv("./data/listings_summary.csv"))
ols.clean_data()
ols.calculate_models()
ols.output_latex()
