import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import BERLIN_CENTER, CORR_MATRIX_COLUMNS, OUTPUT_PATH

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "xelatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

sns.set()


class Plotter:
    def __init__(
        self,
        df,
        corr_matrix_columns=CORR_MATRIX_COLUMNS,
        output_path=OUTPUT_PATH,
    ):
        self.df = df
        self.corr_matrix_columns = corr_matrix_columns
        self.output_path = output_path

    def save_figures(self):
        self.save_center_plot()
        self.save_corr_matrix()
        for col in self.df.columns:
            if self.df[col].dtype == object:
                self.save_occurrence_plot(col)
            else:
                self.save_box_plot(col)

    def save_corr_matrix(self, file_name="CorrMatrix"):
        fig, ax = plt.subplots(figsize=(5, 5))
        corr_frame = self.df[self.corr_matrix_columns]
        sns.heatmap(corr_frame.corr().round(2), annot=True, ax=ax)
        fig.savefig(f"{self.output_path}/{file_name}.pgf", bbox_inches="tight")
        plt.close()

    def save_box_plot(self, col_name, file_name="BoxPlot"):
        fig, ax = plt.subplots(figsize=(4, 4,))
        sns.boxplot(y=self.df[col_name])
        fig.savefig(
            f"{self.output_path}/{col_name}{file_name}.pgf",
            bbox_inches="tight",
        )
        plt.close()

    def save_occurrence_plot(self, col_name, file_name="PieChart"):
        fig, ax = plt.subplots()
        self.df[col_name].value_counts().plot.pie(
            autopct="%1.2f%%",
            ax=ax,
            wedgeprops=dict(linewidth=0),
            shadow=True,
            pctdistance=0.8,
            textprops=dict(color="w", weight="bold"),
        )
        fig.legend(title=col_name)
        fig.savefig(
            f"{self.output_path}/{col_name}{file_name}.pgf",
            bbox_inches="tight",
        )
        plt.close()

    def save_center_plot(self, file_name="DistanceFromCenterPlot"):
        fig, ax = plt.subplots()
        self.df.plot.scatter(
            x="Latitude",
            y="Longitude",
            alpha=0.4,
            figsize=(6, 6),
            c="Price",
            cmap="gist_heat_r",
            colorbar=True,
            sharex=False,
            ax=ax,
        )
        ax.plot(*BERLIN_CENTER, "o", label="Center")
        ax.legend()
        fig.savefig(f"{self.output_path}/{file_name}.pgf", bbox_inches="tight")
        plt.close()
