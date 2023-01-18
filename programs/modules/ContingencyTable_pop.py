""" this file is ContingencyTable class. """
# encoding: utf-8

import numpy as np
import sys
import math


class ContingencyTable:
    def __init__(self, prob_c, prob_e, pop_name, pop_val, sample_size, is_nw, discard_preventive=True, discard_undef_phi=True, discard_undef_dfh=True):
        self.prob_c = prob_c
        self.prob_e = prob_e
        self.pop = pop_val
        if pop_name == 'phi0':
            self.prob_ce = self.__prob_ce_by_phi0(prob_c, prob_e, pop_val)
        else:
            print("invalid population name")
            sys.exit()
        self.sample_size = sample_size
        self.is_nw = is_nw
        self.discard_preventive = discard_preventive
        self.discard_undef_phi = discard_undef_phi
        self.discard_undef_dfh = discard_undef_dfh

        while True:
            self.sampling(is_nw)
            if not self.sampling_rule():
                break
            self.abcd_array = np.zeros(4)

    def sampling(self, is_nw):
        prob_sa = self.prob_ce
        prob_sb = self.prob_c - self.prob_ce
        prob_sc = self.prob_e - self.prob_ce
        prob_sd = 1.0 - self.prob_c - self.prob_e + self.prob_ce
        prob_sampling = [prob_sa, prob_sb, prob_sc, prob_sd]
        ret_array = np.zeros(4)

        sampling_sum = 0
        while sampling_sum < self.sample_size:
            choice_sti = np.random.choice(range(4), p=prob_sampling)
            ret_array[choice_sti] += 1
            if is_nw:
                sampling_sum = np.sum(ret_array[0:3])
            else:
                sampling_sum = np.sum(ret_array)
        self.abcd_array = ret_array

    def sampling_rule(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        do_loop1 = self.discard_preventive and ((a_samp * d_samp) < (b_samp * c_samp))
        do_loop2 = self.discard_undef_phi and (
            ((a_samp + b_samp) == 0)
            or ((c_samp + d_samp) == 0)
            or ((a_samp + c_samp) == 0)
            or ((d_samp + b_samp) == 0)
        )
        do_loop3 = self.discard_undef_dfh and ((a_samp + b_samp == 0) or (a_samp + c_samp == 0))
        return (do_loop1 or do_loop2 or do_loop3)

    def get_pop(self):
        return self.pop

    def __prob_ce_by_phi0(self, prob_c, prob_e, phi0):
        prob_ce = prob_c * prob_e + phi0 * math.sqrt(
            prob_c * (1.0 - prob_c) * prob_e * (1.0 - prob_e))
        return np.round(prob_ce, 15)
