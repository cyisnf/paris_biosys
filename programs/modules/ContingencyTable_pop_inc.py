""" this file is ContingencyTable class. """
# encoding: utf-8

import numpy as np
import sys
import math


class ContingencyTable:
    def __init__(self, prob_c, prob_e, pop_name, pop_val, is_nw):
        self.prob_c = prob_c
        self.prob_e = prob_e
        self.pop = pop_val
        if pop_name == 'phi0':
            self.prob_ce = self.__prob_ce_by_phi0(prob_c, prob_e, pop_val)
        else:
            print("invalid population name")
            sys.exit()

        self.prob_sa = self.prob_ce
        self.prob_sb = self.prob_c - self.prob_ce
        self.prob_sc = self.prob_e - self.prob_ce
        self.prob_sd = 1.0 - self.prob_c - self.prob_e + self.prob_ce
        self.prob_sampling = np.array([self.prob_sa, self.prob_sb, self.prob_sc, self.prob_sd])
        self.is_nw = is_nw

        self.sampling_sum = 0
        self.abcd_array = np.zeros(4)

    def __prob_ce_by_phi0(self, prob_c, prob_e, phi0):
        prob_ce = prob_c * prob_e + phi0 * math.sqrt(
            prob_c * (1.0 - prob_c) * prob_e * (1.0 - prob_e))
        return np.round(prob_ce, 15)

    def sampling_inc(self):
        if self.is_nw:
            choice_sti = np.random.choice(range(3), p=self.prob_sampling[0:3] / np.sum(self.prob_sampling[0:3]))
        else:
            choice_sti = np.random.choice(range(4), p=self.prob_sampling)
        self.abcd_array[choice_sti] += 1

    def def_phi(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        return not (a_samp + b_samp) * (c_samp + d_samp) * (a_samp + c_samp) * (d_samp + b_samp) == 0

    def def_dp(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        return not (a_samp + b_samp) * (c_samp + d_samp) == 0

    def def_dfh(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        return not (a_samp + b_samp) * (a_samp + c_samp) == 0

    def def_prs(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        return not (a_samp == 0) * (b_samp == 0) * (c_samp == 0)
