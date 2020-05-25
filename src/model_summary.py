class ModelSummary:
    def __init__(self, failed_tests, variables):
        self.failed_tests = failed_tests
        self.variables = variables

    def to_string(self):
        stra = "=====================\n"
        stra += " ".join(self.variables)
        stra += "\n======FAILED TESTS======\n"
        for test in self.failed_tests:
            stra += test.name
            if hasattr(test, "pvalue"):
                stra += str(test.pvalue)
            stra += " "
        stra += "\n=====================\n"
        return stra
