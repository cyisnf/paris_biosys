"""This is a models program."""
# encoding: utf-8

import numpy as np
import math
np.seterr(divide='ignore', invalid='ignore')


def convert_array(samples: np.ndarray) -> tuple:
    return (samples[0], samples[1], samples[2], samples[3])


def phi0(contingencyt):
    return contingencyt.get_phi0()


def paris(samples):
    a, b, c, d = np.float64(convert_array(samples))
    res = a / (a + b + c)
    return res


def dfh(samples):
    a, b, c, d = np.float64(convert_array(samples))
    res = a / (math.sqrt((a + b) * (a + c)))
    return res


def phi(samples):
    a, b, c, d = np.float64(convert_array(samples))
    res = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return res


def deltap(samples):
    a, b, c, d = np.float64(convert_array(samples))
    res = (a * d - b * c) / ((a + b) * (c + d))
    return res
