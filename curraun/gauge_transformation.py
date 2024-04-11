import os
# os.environ["MY_NUMBA_TARGET"] = "python"  # Pure Python version    # TODO: remove debug code
# from curraun import su as su
from curraun.numba_target import myjit

import math
import numpy as np

import curraun.su as su

# Constructs the plus links in temporal gauge over the x^+ axis
@myjit
def get_plus_links(ux, ae, z, g, a, N):
    # r = su.mul(su.mexp(su.mul_s(ae, 1j*g*a*(z-N//2))), ux)
    r = su.mul(su.mexp(su.mul_s(ae, z-N//2)), ux)
    return r

# Defines the gauge operator for a given step given its value at the previous one
@myjit
def gauge_transformation_operator(ux, ae, v, z, g, a, N):
    # umin = su.mul(su.mexp(su.mul_s(ae, 1j*g*a*(z-N//2))), su.dagger(ux))
    umin = su.mul(su.mexp(su.mul_s(ae, z-N//2)), su.dagger(ux))
    r = su.mul(umin, v)
    return r

# Defines the gauge operator for a given step given its value at the previous one
@myjit
def act_on_links(u, v1, v2):
    r = su.mul(su.mul(su.dagger(v1), u), v2)
    return r