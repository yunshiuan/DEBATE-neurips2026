"""Preserve the floating-point semantics used by the original Python 3.10 run."""
import math
import statistics


def population_sd(values):
    if len(values) < 2:
        return math.nan
    # Python 3.10 statistics.pstdev rounded the variance to float before taking
    # its square root. Python 3.11 takes an exact-fraction square root instead.
    # Their ~1-ulp differences can split DeGroot SD ties and change rank metrics.
    # Keep the original numerical semantics, without rounding stance inputs or
    # changing the population-SD definition, on both supported Python versions.
    return math.sqrt(statistics.pvariance(values))


def configure():
    import generate_plots
    generate_plots.safe_pstdev = population_sd
