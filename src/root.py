import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_initial = pd.read_csv('./data/listings_summary.csv')

print("The dataset has {} rows and {} columns.".format(*df_initial.shape))
print("It contains {} duplicates.".format(df_initial.duplicated().sum()))
print(df_initial.head(1))
print(df_initial.columns)

kept_columns = ['']