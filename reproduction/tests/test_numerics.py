"""Regression for a floating-point near-tie that affects rank correlations."""
import math
from reproduction.numerics import population_sd
from reproduction.run import load_renderer


def test_original_population_sd_tie_preserves_rank_result():
    # Continuous DeGroot outputs. The first SD rounds to ...6678 under the
    # original run, and to ...668 under Python3.11's revised pstdev algorithm.
    vector=[5.634325294236246,4.365674705763754,1.971621400828484,5.45148794135437]
    tied_sd=1.4594146166646678
    spread=[population_sd(vector),population_sd([-tied_sd,tied_sd]),population_sd([0.,0.])]
    assert spread==[tied_sd,tied_sd,0.]
    load_renderer()
    import generate_plots
    assert math.isclose(generate_plots.spearman_rho([1.,2.,3.],spread),-math.sqrt(3)/2,abs_tol=1e-14)
    # A one-ulp shift breaks that tie and produces a materially different rank.
    split_tie=[math.nextafter(tied_sd,math.inf),tied_sd,0.]
    assert generate_plots.spearman_rho([1.,2.,3.],split_tie)==-1.
