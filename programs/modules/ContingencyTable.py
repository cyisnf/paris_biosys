""" this file is ContingencyTable class. """
# encoding: utf-8

import numpy as np
import math


class ContingencyTable:
    """ this class have env(P(C), P(E), P(C^E),
        sampleN, result of sampling, etc. """

    def __init__(self, prob_c, prob_e, mu_sample_size, is_nw, discard_preventive=True, discard_undef_phi=True, discard_undef_dfh=True):
        self.prob_c = prob_c
        self.prob_e = prob_e
        self.prob_ce = self.__prob_ce(prob_c, prob_e)
        self.mu_sample_size = mu_sample_size
        self.sample_size = self.__sample_size(mu_sample_size)
        self.is_nw = is_nw
        self.discard_preventive = discard_preventive
        self.discard_undef_phi = discard_undef_phi
        self.discard_undef_dfh = discard_undef_dfh

        while True:
            self.abcd_array = np.zeros(4)
            self.sampling(is_nw)
            if not self.sampling_rule():
                break

    def sampling_rule(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        do_loop1 = self.discard_preventive and (
            (a_samp * d_samp) < (b_samp * c_samp)
        )
        do_loop2 = self.discard_undef_phi and (
            ((a_samp + b_samp) == 0)
            or ((c_samp + d_samp) == 0)
            or ((a_samp + c_samp) == 0)
            or ((d_samp + b_samp) == 0)
        )
        do_loop3 = self.discard_undef_dfh and (
            (a_samp + b_samp == 0)
            or (a_samp + c_samp == 0)
        )
        return (do_loop1 or do_loop2 or do_loop3)

    def __prob_ce(self, prob_c, prob_e):
        return np.random.uniform(prob_c * prob_e, min(prob_c, prob_e))

    def __sample_size(self, mu_sample_size):
        res = np.random.normal(mu_sample_size, np.power((mu_sample_size / 7), 1))
        return int(np.round(res))

    def sampling(self, is_nw):
        # nw(TRUE): sampling NW , nw(FALSE):sampling N ##
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

    def get_phi0(self):
        return (self.prob_ce - self.prob_c * self.prob_e) / math.sqrt(self.prob_c * self.prob_e * (1 - self.prob_c) * (1 - self.prob_e))

    def is_preventive(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        return a_samp * d_samp < b_samp * c_samp

    def is_non_generative(self):
        a_samp, b_samp, c_samp, d_samp = self.abcd_array
        return a_samp * d_samp <= b_samp * c_samp
