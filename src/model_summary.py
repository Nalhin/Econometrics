import os
import random

from mako.template import Template

from src.config import OUTPUT_PATH
from src.stat_tests_results import TestType

template_path = "latex/templates/summary.tex"


class ModelSummary:
    def __init__(
        self,
        test_results,
        parameter_names,
        predicted,
        prediction_errors,
        rsquare,
    ):
        self.test_results = test_results
        self.parameter_names = parameter_names
        self.predicted_parameters = predicted
        self.prediction_errors = prediction_errors
        self.rsquare = rsquare

    @property
    def passed_tests(self):
        return sum([t.is_passing for t in self.test_results.values()])

    def to_latex(self):
        parameters = zip(self.parameter_names, self.predicted_parameters[1:])

        model = f"y={self.predicted_parameters[0]}const"
        for name, value in parameters:
            model += f"{value}{name} +"

        model = model[:-1]
        total_tests = len(self.test_results.values())

        template = Template(
            filename=template_path,
            input_encoding="utf-8",
            output_encoding="utf-8",
        )
        output = open(
            os.path.join(
                OUTPUT_PATH,
                f"{self.passed_tests}with{random.randint(1, 99999999999)}.tex",
            ),
            "w",
            encoding="utf-8",
            errors="ignore",
        )
        temp = template.render(
            passed_tests=self.passed_tests,
            total_tests=total_tests,
            model=model,
            rsquare=self.rsquare,
            prediction_errors=self.prediction_errors,
            CATALYSIS_EFFECT=self.test_results[TestType.CATALYSIS_EFFECT],
            SIGNIFICANCE_OF_PARAMETERS=self.test_results[
                TestType.SIGNIFICANCE_OF_PARAMETERS
            ],
            COINCIDENCE=self.test_results[TestType.COINCIDENCE],
            R_SQUARE_SIGNIFICANCE=self.test_results[
                TestType.R_SQUARE_SIGNIFICANCE
            ],
            JARQUE_BERA=self.test_results[TestType.JARQUE_BERA],
            RUNS=self.test_results[TestType.RUNS],
            CHOW=self.test_results[TestType.CHOW],
            COLLINEARITY=self.test_results[TestType.COLLINEARITY],
            BREUCH_GODFREY=self.test_results[TestType.BREUCH_GODFREY],
            BREUCH_PAGAN=self.test_results[TestType.BREUCH_PAGAN],
            RAMSEY_RESET=self.test_results[TestType.RAMSEY_RESET],
        )
        output.write(temp.decode("utf-8"))
        output.close()
