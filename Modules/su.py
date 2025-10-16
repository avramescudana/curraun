"""
Select suitable SU group.
"""
import os

su_group = os.environ.get('GAUGE_GROUP', 'su2').lower()

if su_group == 'su2':
    #print("Using SU(2)")
    from Modules.su2 import *               # Imports everything (*) from Python module curraun.su2
                                            # curraun.su2: A file/python module that defines SU(2) matrix algebra operations
    NC = 2
elif su_group == 'su2_complex':
    #print("Using SU(2) complex")
    from Modules.su2_complex import *
    NC = 2
elif su_group == 'su3':
    #print("Using SU(3)")
    from Modules.su3 import *
    NC = 3
else:
    print("Unsupported gauge group: " + su_group)
    exit()


def _doctests():
    """
    Since SU(2) is a subgroup of SU(3), the compatibility of group functions can be tested:

    >>> import curraun.su2 as su2
    >>> import curraun.su3 as su3

    Perform a calculation in SU(2)

    >>> a2 = su2.get_algebra_element((1.,0.,0.))
    >>> a2
    (0.0, 0.5, 0.0, 0.0)

    >>> b2 = su2.mexp(a2)
    >>> b2
    (0.8775825618903728, 0.479425538604203, 0.0, 0.0)

    >>> c2 = su2.ah(b2)
    >>> c2
    (0.0, 0.479425538604203, 0.0, 0.0)

    >>> su2.dagger(c2)
    (0.0, -0.479425538604203, -0.0, -0.0)

    >>> d2 = su2.sq(c2)
    >>> d2
    0.4596976941318603

    Perform the same calculation in a subgroup of SU(3)

    >>> a3 = su3.get_algebra_element((1,0,0,0,0,0,0,0))
    >>> a3
    (0j, 0.5j, 0j, 0.5j, 0j, 0j, 0j, 0j, 0j)

    >>> b3 = su3.mexp(a3)
    >>> b3
    ((0.8775825618903728+0j), 0.479425538604203j, 0j, 0.479425538604203j, (0.8775825618903728+0j), 0j, 0j, 0j, (1+0j))

    >>> c3 = su3.ah(b3)
    >>> c3
    (0j, 0.479425538604203j, 0j, 0.479425538604203j, 0j, 0j, 0j, 0j, 0j)

    >>> su3.dagger(c3)
    (-0j, -0.479425538604203j, -0j, -0.479425538604203j, -0j, -0j, -0j, -0j, -0j)

    >>> d3 = su3.sq(c3)
    >>> d3
    0.4596976941318603

    Both results should agree

    >>> d2 == d3
    True

    """
    return


if __name__ == "__main__":
    import doctest
    doctest.testmod()
