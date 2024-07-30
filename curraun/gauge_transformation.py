import os
# os.environ["MY_NUMBA_TARGET"] = "python"  # Pure Python version    # TODO: remove debug code
# from curraun import su as su
from curraun.numba_target import myjit

import math
import numpy as np

import curraun.su as su

# Constructs the plus links in temporal gauge over the x^+ axis
def get_plus_links(ux, ae, z, N, t):
    r = su.mul(su.mexp(su.mul_s(ae, (z-N//2)/t**2)), ux)
    return r

# Defines the gauge operator for a given step given its value at the previous one
def gauge_transformation_operator(ux, ae, v, z, N, t):
# def gauge_transformation_operator(ux, ae, v, z, N):
    umin = su.mul(su.mexp(su.mul_s(ae, (z-N//2)/t**2)), ux)
    # umin = su.mul(su.mexp(su.mul_s(ae, z-N//2)), su.dagger(ux))
    r = su.mul(su.dagger(umin), v)
    return r

# Defines the gauge operator for a given step given its value at the previous one
def act_on_links(u_temp, v1, v2):
    r = su.mul(su.mul(su.dagger(v1), u_temp), v2)
    return r