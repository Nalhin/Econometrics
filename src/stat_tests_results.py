from abc import abstractmethod, ABC
from enum import Enum

from .config import P_VALUE_THRESHOLD


class TestResult(ABC):
    def __init__(self, type):
        self.type = type

    @property
    @abstractmethod
    def is_passing(self):
        return NotImplemented


class PValueTestResult(TestResult):
    def __init__(
        self, type, p_value, test_stat_value, smaller=False, optional=False,
    ):
        super().__init__(type)
        self.p_value = p_value
        self.test_stat_value = test_stat_value
        self.smaller = smaller
        self.optional = optional

    @property
    def is_passing(self):
        if self.optional:
            return True
        if self.smaller:
            return self.p_value < P_VALUE_THRESHOLD
        return self.p_value > P_VALUE_THRESHOLD


class CatalysisTestResult(TestResult):
    def __init__(self, catalysis_pairs):
        super().__init__(type=TestType.CATALYSIS_EFFECT)
        self.catalysis_pairs = catalysis_pairs

    @property
    def is_passing(self):
        return not len(self.catalysis_pairs)


class SignificanceOfParametersTestResult(TestResult):
    def __init__(self, variables):
        super().__init__(type=TestType.SIGNIFICANCE_OF_PARAMETERS)
        self.variables = variables

    @property
    def is_passing(self):
        return all(p["p_value"] < P_VALUE_THRESHOLD for p in self.variables)


class CoincidenceTestResult(TestResult):
    def __init__(self, coincidence_errors, summary):
        super().__init__(type=TestType.COINCIDENCE)
        self.coincidence_errors = coincidence_errors
        self.summary = summary

    @property
    def is_passing(self):
        return not len(self.coincidence_errors)


class CollinearityTestResult(TestResult):
    def __init__(self, variables):
        super().__init__(TestType.COLLINEARITY)
        self.collinear_variables = variables

    @property
    def is_passing(self):
        return not len(self.collinear_variables)


class Estimator:
    def __init__(self, me, mae, rmse, mape):
        self.me = me
        self.mae = mae
        self.rmse = rmse
        self.mape = mape


class TestType(Enum):
    CATALYSIS_EFFECT = 1
    SIGNIFICANCE_OF_PARAMETERS = 2
    COINCIDENCE = 3
    R_SQUARE_SIGNIFICANCE = 4
    JARQUE_BERA = 5
    RUNS = 6
    CHOW = 7
    COLLINEARITY = 8
    BREUCH_GODFREY = 9
    BREUCH_PAGAN = 10
    RAMSEY_RESET = 11
