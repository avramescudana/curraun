from curraun.numba_target import myjit
import curraun.su as su

GRADIENT_ITERATION_MAX = 1000 # 30 # 25000 # 2500 # 250 # 2500 # 50 # 2500000 # 30  # 25000 # 30 # 250
GRADIENT_ITERATION_BOUND = 1e-8 #20 #su.EXP_ACCURACY_SQUARED

HEAVY_BALL_BETA = 0.9

# Gradient descent on algebra element
# Calculate gradient analytically
@myjit
def init_kernel_2_su3(xi, u0, u1, ua, ub):
    #if xi < 750:
    #    return
    # if xi != 594 and xi != 750 and xi != 814:
    #     return
    # if xi == 10:
    #     exit()
    # initialize transverse gauge links (longitudinal magnetic field)
    # (see PhD thesis eq.(2.135))  # TODO: add proper link or reference
    for d in range(2):
        b1 = su.load(ua[xi, d])
        b1 = su.add(b1, ub[xi, d])  # A

        m1 = su.zero_algebra  # real values

        # Better starting value:
        m1a = su.get_algebra_factors_from_group_element_approximate(ua[xi, d])
        m1b = su.get_algebra_factors_from_group_element_approximate(ub[xi, d])
        m1 = su.add_algebra(m1a, m1b)
        # m1 = su.mul_algebra(m1, 2)
        #m1 = su.add_algebra(mul_algebra(m1a, -1), mul_algebra(m1b, -1)) # ++
        #m1 = su.add_algebra(mul_algebra(m1a, 0), mul_algebra(m1b, 0)) # --
        #m1 = su.add_algebra(mul_algebra(m1a, 2), mul_algebra(m1b, 2)) # ++
        m1_prev = m1

        # Make solution consistently unitary
        epsilon2 = 0.5 # 0.125 # 0.0001 # 0.125

        si3 = 0
        si4 = 0

        smallestloss = 1

        for i in range(GRADIENT_ITERATION_MAX):
            # Calculate Loss:
            b3 = su.mexp(su.get_algebra_element(m1))
            loss1, check1, check2, check3 = loss(b1, b3)

            m2new = m1

            epsilon1 = epsilon2 # * 0.125 # smaller
            epsilon1 = 1.e-8

            # Calculate analytic derivative:
            grad = gradient(b1, m1)

            m2new = su.add_algebra(m2new, su.mul_algebra(grad, -epsilon2))

            b3new = su.mexp(su.get_algebra_element(m2new))

            # Heavy ball method
            # dball = + beta * (m1 - m1_prev)
            dball = su.add_algebra(m1, su.mul_algebra(m1_prev, -1))
            dball = su.mul_algebra(dball, 1) # 0.6) # epsilon2)

            m2new21 = su.add_algebra(m2new, dball)

            b3new21 = su.mexp(su.get_algebra_element(m2new21))


            loss3, check4, check5, check6 = loss(b1, b3new)
            loss21 = loss(b1, b3new21)[0]

            m1_prev = m1

            # Find step with smallest value of loss
            smallestloss_prev = smallestloss
            smallestloss = loss1
            smallestitem = -1
            if loss3 < smallestloss:
                m1 = m2new
                smallestloss = loss3
                smallestitem = 5
            if loss21 < smallestloss:
                m1 = m2new21
                smallestloss = loss21
                smallestitem = 21

            if smallestitem == 5:
                si3 +=1
            if smallestitem == 21:
                si4 +=1

            if smallestitem == -1:
                pass


            if smallestloss < GRADIENT_ITERATION_BOUND:
            #    if debug: # TODO: Remove debugging code
            #        print("Kernel 2: {} iterations: {}".format(i, loss3))
            #    print("Kernel 2: xi:", xi, ", d:", d, ": Iterations:", i, ". Bounds:", loss3)
            #    print("Kernel 2: xi:", xi, ", d:", d, ": Iterations:", i, ". Bounds:", loss3, ", eps: ", epsilon2, (si3, si4))
                break
        else: # no break
            # pass
        #    if debug:
        #    print("Kernel 2: xi:", xi, ", d:", d, ": max iterations reached:", i, ". Bounds:", loss3)
            print("Kernel 2: xi:", xi, ", d:", d, ": max iterations reached:", i, ". Bounds:", loss3, ", eps: ", epsilon2, (si3, si4))
        #    print("Kernel 2: max iterations reached. bounds: {}".format(loss3))
        #        print("xi: {}, d: {}".format(xi, d))

        su.store(u0[xi, d], b3)
        su.store(u1[xi, d], b3)

@myjit
def gradient(b1, m1):
    grad = su.zero_algebra
    for a in range(8):
        mdelta = su.unit_algebra[a]

        # Calculate analytic derivative in direction mdelta
        grad_a = gradient_component(b1, m1, mdelta)
        grad = su.add_algebra(grad, su.mul_algebra(mdelta, grad_a))
    return grad

@myjit
def gradient_component(b1, m1, mdelta):

    # b1 # A
    unit = su.unit()
    b3 = su.mexp(su.get_algebra_element(m1)) # B = exp(m1)

    e1 = su.mul(b1, su.dagger(su.add(unit, b3))) # X = A (1 + B)^dagger

    e2 = su.add(e1, su.mul_s(su.dagger(e1), -1)) # X - X^dagger
    e3 = su.mul_s(unit, -su.tr(e2) / su.N_C) # - tr(e2) / N_C * 1
    e4 = su.add(e2, e3) # Y = [X]_ah

    m3 = su.get_algebra_element(mdelta)

    db3 = su.dmexp(su.get_algebra_element(m1), su.mul_s(m3, -1)) # derivative of b3
    f1 = su.mul(b1, su.dagger(db3))  # A dB^dagger

    f2 = su.add(f1, su.mul_s(su.dagger(f1), -1)) # f1 - f1^dagger
    f3 = su.mul_s(unit, -su.tr(f2) / su.N_C) # - tr(f2) / N_C * 1
    f4 = su.add(f2, f3) # d/da Y = d/da [X]_ah

    g1 = su.mul(f4, su.dagger(e4)) # (d/da Y) . Y^dagger
    g2 = su.mul(e4, su.dagger(f4)) # Y . d/da Y^dagger
    g3 = su.tr(su.add(g1, g2)) # d/da tr(Y . Y^dagger) = tr(g1 + g2)

    res = -g3.real * 0.25 # Result should be real
    return res

@myjit
def loss(b1, b3):
    unit = su.unit()

    # Check result
    e1 = su.mul(b1, su.dagger(su.add(unit, b3)))
    e2 = su.ah(e1)
    check1 = su.sq(e2)

    # Check unitarity:
    check2 = su.check_unitary(b3)

    # Check determinant
    f1 = su.det(b3) - 1
    check3 = f1.real * f1.real + f1.imag * f1.imag

    #res = check1 + check2 + check3
    res = check1

    return res, check1, check2, check3
