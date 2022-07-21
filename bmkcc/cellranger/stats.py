#!/usr/bin/env python
#
# Copyright (c) 2015 10X Genomics, Inc. All rights reserved.
#

from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import scipy.stats as sp_stats
from scipy.special import loggamma
from six.moves import xrange

import bmkcc.tenkit.stats as tk_stats


def effective_diversity(counts):
    """ Inverse Simpson Index, or the effective diversity of power 2 """
    numerator = np.sum(counts) ** 2
    denominator = np.sum(v ** 2 for v in counts)
    return tk_stats.robust_divide(float(numerator), float(denominator))


def eval_multinomial_loglikelihoods(matrix, profile_p, max_mem_gb=0.1):
    """Compute the multinomial log PMF for many barcodes
    Args:
      matrix (scipy.sparse.csc_matrix): Matrix of UMI counts (feature x barcode)
      profile_p (np.ndarray(float)): Multinomial probability vector
      max_mem_gb (float): Try to bound memory usage.
    Returns:
      log_likelihoods (np.ndarray(float)): Log-likelihood for each barcode
    """
    gb_per_bc = float(matrix.shape[0] * matrix.dtype.itemsize) / (1024 ** 3)
    bcs_per_chunk = max(1, int(np.round(max_mem_gb / gb_per_bc)))
    num_bcs = matrix.shape[1]

    loglk = np.zeros(num_bcs)

    for chunk_start in xrange(0, num_bcs, bcs_per_chunk):
        chunk = slice(chunk_start, chunk_start + bcs_per_chunk)
        matrix_chunk = matrix[:, chunk].transpose().toarray()
        n = matrix_chunk.sum(1)
        loglk[chunk] = sp_stats.multinomial.logpmf(matrix_chunk, n, p=profile_p)
    return loglk


def simulate_multinomial_loglikelihoods(
    profile_p, umis_per_bc, num_sims=1000, jump=1000, n_sample_feature_block=1000000, verbose=False
):
    """Simulate draws from a multinomial distribution for various values of N.

       Uses the approximation from Lun et al. ( https://www.biorxiv.org/content/biorxiv/early/2018/04/04/234872.full.pdf )

    Args:
      profile_p (np.ndarray(float)): Probability of observing each feature.
      umis_per_bc (np.ndarray(int)): UMI counts per barcode (multinomial N).
      num_sims (int): Number of simulations per distinct N value.
      jump (int): Vectorize the sampling if the gap between two distinct Ns exceeds this.
      n_sample_feature_block (int): Vectorize this many feature samplings at a time.
    Returns:
      (distinct_ns (np.ndarray(int)), log_likelihoods (np.ndarray(float)):
      distinct_ns is an array containing the distinct N values that were simulated.
      log_likelihoods is a len(distinct_ns) x num_sims matrix containing the
        simulated log likelihoods.
    """
    np.random.seed(0)

    distinct_n = np.flatnonzero(np.bincount(umis_per_bc))

    loglk = np.zeros((len(distinct_n), num_sims), dtype=float)
    num_all_n = np.max(distinct_n) - np.min(distinct_n)
    if verbose:
        print("Number of distinct N supplied: %d" % len(distinct_n))
        print("Range of N: %d" % num_all_n)
        print("Number of features: %d" % len(profile_p))

    sampled_features = np.random.choice(
        len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True
    )
    k = 0

    log_profile_p = np.log(profile_p)

    for sim_idx in xrange(num_sims):
        if verbose and sim_idx % 100 == 99:
            sys.stdout.write(".")
            sys.stdout.flush()
        curr_counts = np.ravel(sp_stats.multinomial.rvs(distinct_n[0], profile_p, size=1))

        curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[0], p=profile_p)

        loglk[0, sim_idx] = curr_loglk

        for i in xrange(1, len(distinct_n)):
            step = distinct_n[i] - distinct_n[i - 1]
            if step >= jump:
                # Instead of iterating for each n, sample the intermediate ns all at once
                curr_counts += np.ravel(sp_stats.multinomial.rvs(step, profile_p, size=1))
                curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[i], p=profile_p)
                assert not np.isnan(curr_loglk)
            else:
                # Iteratively sample between the two distinct values of n
                for n in xrange(distinct_n[i - 1] + 1, distinct_n[i] + 1):
                    j = sampled_features[k]
                    k += 1
                    if k >= n_sample_feature_block:
                        # Amortize this operation
                        sampled_features = np.random.choice(
                            len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True
                        )
                        k = 0
                    curr_counts[j] += 1
                    curr_loglk += log_profile_p[j] + np.log(float(n) / curr_counts[j])

            loglk[i, sim_idx] = curr_loglk

    if verbose:
        sys.stdout.write("\n")

    return distinct_n, loglk


def compute_ambient_pvalues(umis_per_bc, obs_loglk, sim_n, sim_loglk):
    """Compute p-values for observed multinomial log-likelihoods
    Args:
      umis_per_bc (nd.array(int)): UMI counts per barcode
      obs_loglk (nd.array(float)): Observed log-likelihoods of each barcode deriving from an ambient profile
      sim_n (nd.array(int)): Multinomial N for simulated log-likelihoods
      sim_loglk (nd.array(float)): Simulated log-likelihoods of shape (len(sim_n), num_simulations)
    Returns:
      pvalues (nd.array(float)): p-values
    """
    assert len(umis_per_bc) == len(obs_loglk)
    assert sim_loglk.shape[0] == len(sim_n)

    # Find the index of the simulated N for each barcode
    sim_n_idx = np.searchsorted(sim_n, umis_per_bc)
    num_sims = sim_loglk.shape[1]

    num_barcodes = len(umis_per_bc)

    pvalues = np.zeros(num_barcodes)

    for i in xrange(num_barcodes):
        num_lower_loglk = np.sum(sim_loglk[sim_n_idx[i], :] < obs_loglk[i])
        pvalues[i] = float(1 + num_lower_loglk) / (1 + num_sims)
    return pvalues


class Curve(object):
    """Curve information for plotting

    Attributes
        x           (list of int): Curve x-axis (number of cells)
        y           (list of float): Curve y-axis (expected value of number of
                    unique clonotypes)
        y_std       (list of float): Standard deviation of number of unique
                    clonotypes in a curve
        y_ciu       (list of float): Upper bound of 95% confidence interval
                    for number of unique clonotypes in a curve
        y_cil       (list of float): Lower bound of 95% confidence interval
                    for number of unique clonotypes in a curve
    """

    def __init__(self):
        """Initialize an empty curve"""
        self.x = []
        self.y = []
        self.y_std = []
        self.y_ciu = []
        self.y_cil = []

    def is_empty(self):
        """Check if curve is empty or calculated

        Args:
            None
        Returns:
            bool
        """
        if len(self.y) == 0 or len(self.x) == 0:
            return True
        return False

    def is_consistent(self):
        """Check if calculated curve is consistent (length match)

        Args:
            None
        Returns:
            bool
        """
        len_list = [len(i) for i in [self.x, self.y, self.y_cil, self.y_ciu]]
        if len_list.count(len_list[0]) != len(len_list):
            return False
        return True


class Diversity(object):
    """Represents diversity with rarefaction and extrapolation curves.
    Attributes:
        sorted_hist         (list of tuple): A histogram sorted by abundance
                                            (most abundant has lowest index)
        freq_counts         (dict of int: int): Frequency counts table
        N                   (int): total number of counts in histogram
        rarefaction_curve   (Curve): Rarefaction curve information
        extrapolation_curve (Curve): Extrapolation curve information
    """

    def __init__(self, hist):
        self.sorted_hist = sorted(hist, reverse=True)
        self.N = self._calc_n()
        self.freq_counts = self._get_freq_counts()
        # Rarefaction curve
        self.rarefaction_curve = Curve()
        # Extrapolation curve
        self.extrapolation_curve = Curve()

    def is_diversity_curve_possible(self):
        if self.N == 0 or self.f_0_chao1() in [0, -1]:
            return False
        return True

    def _get_freq_counts(self):
        """Get frequency counts (f_k in Colwell et al 2012)

        Args:
            None

        Returns:
            dict
        """
        ret_dict = {}
        for abund in self.sorted_hist:
            if abund not in ret_dict:
                ret_dict[abund] = 1
            else:
                ret_dict[abund] += 1
        return ret_dict

    def _calc_n(self):
        """Calculates number of samples. (length of input histogram)

        Args:
            None

        Returns:
            int
        """
        return sum(self.sorted_hist)

    def _alpha(self, n, k):
        """Calculates alpha parameter in Colwell et al 2012

        Args:
            n (int): rarefaction point n
            k (int): Frequency

        Returns:
            int
        """
        if k > self.N - n:
            return 0
        log_alpha = (
            loggamma(self.N - k + 1)
            + loggamma(self.N - n + 1)
            - loggamma(self.N + 1)
            - loggamma(self.N - k - n + 1)
        )
        return np.exp(np.real(log_alpha))

    # Minimum variance unbiased estimator (MVUE) for both hypergeometric
    # and multinomial models
    # eq (4) in Colwell et al 2012
    # eq (5) in Colwell et al 2012
    def _rarefaction(self, n):
        """Calculates rarefaction and standard deviation of rarefaction for a single point n

        Args:
            n (int): rarefaction point n

        Returns:
            int, int
        """
        # number of observed clonotypes (S_obs in paper)
        num_clon = float(sum(self.freq_counts.values()))
        sum_helper_exp_val = 0
        for k in self.freq_counts:
            sum_helper_exp_val += self._alpha(n, k) * self.freq_counts[k]
        exp_val = num_clon - sum_helper_exp_val
        sum_helper_std_dev = 0
        for k in self.freq_counts:
            sum_helper_std_dev += (1 - self._alpha(n, k)) ** 2 * self.freq_counts[k] - float(
                exp_val ** 2
            ) / self.assemblage_size_estimate()
        return exp_val, np.sqrt(sum_helper_std_dev)

    def calc_rarefaction_curve(self, num_steps=40):
        """Calculates rarefaction curve for num_steps between 1 and N

        Args:
            num_steps (int): Number of steps between 1 and N

        Returns:
            None
        """
        step_size = int(self.N // (num_steps - 1))
        if step_size == 0:
            step_size = 1
        rc_x = list(range(1, self.N, step_size))
        if len(rc_x) < num_steps:
            rc_x = list(range(1, self.N + step_size, step_size))

        rc_y_exp = [None] * len(rc_x)
        rc_y_std = [None] * len(rc_x)
        rc_y_ciu = [None] * len(rc_x)
        rc_y_cil = [None] * len(rc_x)
        for (idx, n) in enumerate(rc_x):
            rc_y_exp[idx], rc_y_std[idx] = self._rarefaction(n)
            rc_y_cil[idx] = rc_y_exp[idx] - 1.96 * rc_y_std[idx]
            rc_y_ciu[idx] = rc_y_exp[idx] + 1.96 * rc_y_std[idx]
        self.rarefaction_curve.x = rc_x
        self.rarefaction_curve.y = rc_y_exp
        self.rarefaction_curve.y_std = rc_y_std
        self.rarefaction_curve.y_cil = rc_y_cil
        self.rarefaction_curve.y_ciu = rc_y_ciu

    def calc_rarefaction_curve_plotly(self, origin, color, num_steps=40, stdev=True):
        """Create a plotly curve for rarefaction.

        Args:
            origin (str): Origin string (see vdj inputs)
            color: color of the curve
            num_steps (int): number of extrapolation steps
            stdev (bool): flag for calculation of standard deviation (and confidence intarvals)


        Returns:
            [dict]
        """
        self.calc_rarefaction_curve(num_steps)
        lines = []
        lines.append(
            {
                "x": self.rarefaction_curve.x,
                "y": self.rarefaction_curve.y,
                "type": "scatter",
                "name": origin,
                "mode": "lines",
                "line_color": color,
            }
        )
        if stdev:
            lines.append(
                {
                    "x": self.rarefaction_curve.x + self.rarefaction_curve.x[::-1],
                    "y": self.rarefaction_curve.y_ciu + self.rarefaction_curve.y_cil[::-1],
                    "type": "scatter",
                    "name": origin + " (Stdev)",
                    "fill": "toself",
                    "line_color": "rgba(255,255,255,0)",
                    "fillcolor": color,
                    "opacity": 0.2,
                }
            )
        return lines

    # eq (9) in Colwell et al 2012
    # not using the approximation used in eq (9)
    def _extrapolation(self, n_plus_m):
        """Calculates extrapolation for a single point (N + m)
        standard deviation is not implemented.

        Args:
            n_plus_m (int): Distance of extrapolation point to N
                (how many more samples collected)

        Returns:
            int, int
        """
        # number of observed clonotypes (s_obs in paper)
        s_obs = float(sum(self.freq_counts.values()))
        f_0 = float(self.f_0_chao1())
        f_1 = float(self.freq_counts[1])

        N = float(self.N)  # n in paper
        m = float(n_plus_m - self.N)

        brackets = 1.0 - (1.0 - f_1 / N / f_0) ** m

        exp_val = s_obs + f_0 * brackets
        # Standard deviation for extrapolation is not implemented
        std_dev = None

        return exp_val, std_dev

    def calc_extrapolation_curve(self, num_steps=40, max_n=2000):
        """Calculates extrapolation curve for num_steps between N and max_n

        Args:
            num_steps (int): Number of steps between N and max_n
            max_n (int): The limit to which the function extrapolates

        Returns:
            None
        """
        step_size = int((max_n - self.N) / (num_steps - 1))
        if len(list(range(self.N, max_n, step_size))) < num_steps:
            ec_x = list(range(self.N, max_n + step_size, step_size))
        else:
            ec_x = list(range(self.N, max_n, step_size))
        ec_y_exp = [None] * len(ec_x)
        ec_y_std = [None] * len(ec_x)
        ec_y_ciu = [None] * len(ec_x)
        ec_y_cil = [None] * len(ec_x)
        for (idx, n_plus_m) in enumerate(ec_x):
            ec_y_exp[idx], ec_y_std[idx] = self._extrapolation(n_plus_m)
            if ec_y_std[idx] is not None:
                ec_y_cil[idx] = ec_y_exp[idx] - 1.96 * ec_y_std[idx]
                ec_y_ciu[idx] = ec_y_exp[idx] + 1.96 * ec_y_std[idx]
        self.extrapolation_curve.x = ec_x
        self.extrapolation_curve.y = ec_y_exp
        self.extrapolation_curve.y_std = ec_y_std
        self.extrapolation_curve.y_cil = ec_y_cil
        self.extrapolation_curve.y_ciu = ec_y_ciu

    def f_0_chao1(self):
        """Estimate f0 (number of clonotypes that we haven't seen yet) using chao1 estimator
        Based on: Colwell et al 2012

        Args:
            None

        Returns:
            float
        """
        if 1 in self.freq_counts and 2 in self.freq_counts:
            return float(self.freq_counts[1] ** 2) / float(2 * self.freq_counts[2])
        elif 1 in self.freq_counts:
            return float(self.freq_counts[1] * (self.freq_counts[1] - 1)) / 2.0
        else:
            # print "Error in f_0_chao1" # TODO log message
            return -1.0

    def assemblage_size_estimate(self):
        """Estimate assemblage size (number of total unique clonotypes) using chao1 estimator
        Based on: Colwell et al 2012

        Args:
            None

        Returns:
            float
        """
        return self.N + self.f_0_chao1()

    def create_both_curves(self, outfile, rc_steps=40, ec_steps=40, max_n=2000):
        """Calculate rarefaction and extrapolation diversity curves and save them
        in csv format in outfile.

        Args:
            outfile (str): Path to output file
            rc_steps (int): Rarefaction curve steps between 1 and N
            ec_steps (int): Extrapolation steps between N and max_n
            max_n (int): Maximum number of cells extrapolating to

        Returns:
            None
        """
        # Check if extrapolation_curve is calculated. If not, call the appropriate func to calc
        if self.extrapolation_curve.is_empty():
            self.calc_extrapolation_curve(ec_steps, max_n)
        # Check if rarefactoion_curve is calculated. If not, call the appropriate func to calc
        if self.rarefaction_curve.is_empty():
            self.calc_rarefaction_curve(rc_steps)
        if ~self.rarefaction_curve.is_consistent():
            return None  # TODO add appropriate logging message
        if ~self.extrapolation_curve.is_consistent():
            return None  # TODO add appropriate logging message
        header = "x,y,y_cil,y_ciu,type\n"
        with open(outfile, "w") as outhandle:
            outhandle.write(header)
            for (idx, x) in enumerate(self.rarefaction_curve.x):
                numbers = [
                    x,
                    self.rarefaction_curve.y[idx],
                    self.rarefaction_curve.y_cil[idx],
                    self.rarefaction_curve.y_ciu[idx],
                ]
                line = ",".join([str(n) for n in numbers] + ["rarefaction"]) + "\n"
                outhandle.write(line)
            for (idx, x) in enumerate(self.extrapolation_curve.x):
                numbers = [
                    x,
                    self.extrapolation_curve.y[idx],
                    self.extrapolation_curve.y_cil[idx],
                    self.extrapolation_curve.y_ciu[idx],
                ]
                line = ",".join([str(n) for n in numbers] + ["extrapolation"]) + "\n"
                outhandle.write(line)
        return 1
