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
    def __init__(self, catalysis):
        super().__init__("Catalysis effect")
        self.catalysis = catalysis

    @property
    def is_passing(self):
        return not len(self.catalysis)


class SignificanceOfVariablesTestResult(TestResult):
    def __init__(self, variables):
        super().__init__(name="Significance of variables")
        self.variables = variables

    @property
    def is_passing(self):
        for var in self.variables:
            if var["pvalue"] < P_VALUE_THRESHOLD:
                return False
        return True


class CoincidenceTestResult(TestResult):
    def __init__(self, coincidence):
        super().__init__(name="Coincidence")
        self.coincidence = coincidence

    @property
    def is_passing(self):
        for var in self.coincidence:
            if not var["passing"]:
                return False

        return True


class CustomTestResult(TestResult):
    def __init__(self, name, successful):
        super().__init__(name)
        self.successful = successful

    @property
    def is_passing(self):
        return self.successful