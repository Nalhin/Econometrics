from abc import abstractmethod, ABC

from .config import P_VALUE_THRESHOLD


class TestResult(ABC):
    def __init__(self, name):
        self.name = name

    @property
    @abstractmethod
    def is_passing(self):
        return NotImplemented


class PValueTestResult(TestResult):
    def __init__(self, name, pvalue):
        super().__init__(name)
        self.pvalue = pvalue

    @property
    def is_passing(self):
        return self.pvalue > P_VALUE_THRESHOLD


class CatalysisTestResult(TestResult):
    def __init__(self, catalysis_pairs):
        super().__init__("Catalysis effect")
        self.catalysis_pairs = catalysis_pairs

    @property
    def is_passing(self):
        return not len(self.catalysis_pairs)


class SignificanceOfVariablesTestResult(TestResult):
    def __init__(self, variables):
        super().__init__(name="Significance of variables")
        self.variables = variables

    @property
    def is_passing(self):
        return all(p["pvalue"] < P_VALUE_THRESHOLD for p in self.variables)


class CoincidenceTestResult(TestResult):
    def __init__(self, coincidence_errors):
        super().__init__(name="Coincidence")
        self.coincidence_errors = coincidence_errors

    @property
    def is_passing(self):
        return len(self.coincidence_errors)


class CustomTestResult(TestResult):
    def __init__(self, name, successful):
        super().__init__(name)
        self.successful = successful

    @property
    def is_passing(self):
        return self.successful
