import os

from mako.template import Template

from .config import OUTPUT_PATH
from .translations import apply_translations


class Summarizer:
    def __init__(
        self,
        df,
        template_path="latex/templates/table.tex",
        output_path=OUTPUT_PATH,
        default_table_name="Table",
    ):
        self.df = df
        self.template_path = template_path
        self.output_path = output_path
        self.default_table_name = default_table_name

    def generate_summary_stats(self,):
        for col_name in self.df.columns:
            description = self.df[col_name].describe()
            if self.df[col_name].dtype == object:
                self.enrich_object_description(description, self.df[col_name])
            else:
                self.enrich_number_description(description, self.df[col_name])
                self.generate_tex(description, col_name)

    def generate_tex(self, description, col_name):
        template = Template(filename=self.template_path)
        tab = open(
            os.path.join(
                self.output_path, f"{col_name}{self.default_table_name}.tex"
            ),
            "w",
        )
        tab.write(
            template.render(
                title=col_name, column=apply_translations(description).items()
            )
        )
        tab.close()

    @staticmethod
    def enrich_object_description(description, col):
        return description

    @staticmethod
    def enrich_number_description(description, col):
        description["kurtosis"] = col.kurtosis()
        description["skewness"] = col.skew()
        description["std"] = col.std()
        del description["count"]
        return description
