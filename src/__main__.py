import pandas as pd

from .ols_model import OLS

ols = OLS(pd.read_csv("../data/listings_summary.csv"))

ols.clean_data()
ols.calculate_models()

if __name__ == "__main__":
    ols.output_latex()
